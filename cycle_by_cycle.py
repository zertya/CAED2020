# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:30:10 2020

@author: Andreas De Bleser

cycle-by-cycle beta burst analysis of Brainstorm dataset

MNEtools & bycycle
"""
#import
import os
import os.path as op
import pandas as pd
import numpy as np

import mne
from mne import combine_evoked
from mne.minimum_norm import apply_inverse
from mne.datasets.brainstorm import bst_auditory
from mne.io import read_raw_ctf

print(__doc__)

#%% prepare for reading data
#set working director
os.chdir('D:/school/2e master/case studies/CAED2020')
#use precomputed steps
use_precomputed = True 

#get the data in directory/download if not in the directory 
data_path = bst_auditory.data_path('D:/school/2e master/case studies/CAED2020/mne_data')


subject = 'bst_auditory'
subjects_dir = op.join(data_path, 'subjects')

raw_fname1 = op.join(data_path, 'MEG', 'bst_auditory', ##data file of first run
                     'S01_AEF_20131218_01.ds')
raw_fname2 = op.join(data_path, 'MEG', 'bst_auditory', ##data file of scnd run
                     'S01_AEF_20131218_02.ds')
erm_fname = op.join(data_path, 'MEG', 'bst_auditory',  ##noise recording of room 
                    'S01_Noise_20131218_01.ds')
#read data
preload = not use_precomputed ##data stored in memory (needed for some functions)
raw = read_raw_ctf(raw_fname1, preload=preload) 
n_times_run1 = raw.n_times
mne.io.concatenate_raws([raw, read_raw_ctf(raw_fname2, preload=preload)])##make one continious data set out of first and second run
raw_erm = read_raw_ctf(erm_fname, preload=preload)

#define channels
raw.set_channel_types({'HEOG': 'eog', 'VEOG': 'eog', 'ECG': 'ecg'})
if not use_precomputed:
    # Leave out the two EEG channels for easier computation of forward.
    raw.pick(['meg', 'stim', 'misc', 'eog', 'ecg'])

#read bad segments from csv file 
annotations_df = pd.DataFrame()
offset = n_times_run1
for idx in [1, 2]:
    csv_fname = op.join(data_path, 'MEG', 'bst_auditory',
                        'events_bad_0%s.csv' % idx)
    df = pd.read_csv(csv_fname, header=None,
                     names=['onset', 'duration', 'id', 'label'])
    print('Events from run {0}:'.format(idx))
    print(df)

    df['onset'] += offset * (idx - 1)
    annotations_df = pd.concat([annotations_df, df], axis=0)

saccades_events = df[df['label'] == 'saccade'].values[:, :3].astype(int)

# Conversion from samples to time (s):
onsets = annotations_df['onset'].values / raw.info['sfreq']
durations = annotations_df['duration'].values / raw.info['sfreq']
descriptions = annotations_df['label'].values

annotations = mne.Annotations(onsets, durations, descriptions)
raw.set_annotations(annotations)
del onsets, durations, descriptions

#add EOG & saccade projectors
saccade_epochs = mne.Epochs(raw, saccades_events, 1, 0., 0.5, preload=True,
                            baseline=(None, None),
                            reject_by_annotation=False)

projs_saccade = mne.compute_proj_epochs(saccade_epochs, n_mag=1, n_eeg=0,
                                        desc_prefix='saccade')
if use_precomputed:
    proj_fname = op.join(data_path, 'MEG', 'bst_auditory',
                         'bst_auditory-eog-proj.fif')
    projs_eog = mne.read_proj(proj_fname)[0]
else:
    projs_eog, _ = mne.preprocessing.compute_proj_eog(raw.load_data(),
                                                      n_mag=1, n_eeg=0)
raw.add_proj(projs_saccade)
raw.add_proj(projs_eog)
del saccade_epochs, saccades_events, projs_eog, projs_saccade  # To save memory
raw.plot()
#%%epoching & averaging

#add events
events = mne.find_events(raw, stim_channel = "UPPT001")
event_dict = {'standard': 1, 'deviant': 2}
##adjust event timing:difference between auditory channel & stimulus channel 
sound_data = raw[raw.ch_names.index('UADC001-4408')][0][0]
onsets = np.where(np.abs(sound_data) >2. *np.std(sound_data))[0]
min_diff = int(0.5 *raw.info['sfreq'])
diffs = np.concatenate([[min_diff+1],np.diff(onsets)])
onsets = onsets[diffs> min_diff]
assert len(onsets) == len(events)
diffs = 1000. * (events[:, 0] - onsets) / raw.info['sfreq']
print('Trigger delay removed (μ ± σ): %0.1f ± %0.1f ms'
      % (np.mean(diffs), np.std(diffs)))
events[:, 0] = onsets
del sound_data, diffs

#remove noisy channels
raw.info['bads']= ['MLO52-4408','MRT51-4408','MLO42-4408','MLO43-4408']
##visual inspection of bad channels, mark as bad by clicking 
raw.plot(events, event_color = {1: 'b', 2: 'r'})

#epoch
tmin,tmax = -0.1, 0.5 #start & end of time window used to reject epoch 
reject = dict(mag=4e-12, eog=250e-6) #rejection parameters of peak to peak amplitude 
epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, picks=['meg', 'eog'],
                    baseline=(None, 0), reject=reject, preload=False, ##epochs overlapping 'bad' segments removed 
                    proj=True)
epochs.drop_bad()
epochs_standard = mne.concatenate_epochs([epochs['standard'][range(40)], #only 40 good epochs from each run
                                          epochs['standard'][182:222]]) #first epoch from 2nd run corresponds to 182
epochs_standard.load_data()  # Resampling to save memory.
epochs_standard.resample(600, npad='auto')
epochs_deviant = epochs['deviant'].load_data()
epochs_deviant.resample(600, npad='auto')
del epochs

evoked_std = epochs_standard.average()
evoked_dev = epochs_deviant.average()
del epochs_standard, epochs_deviant

#%%preprocessing (normally during RAW stage but to save memory load in evoked stage)
for evoked in (evoked_std, evoked_dev): #lowpass filter of 40Hz to remove power line artifacts
     evoked.filter(l_freq=None, h_freq=40., fir_design='firwin')
#visual inspection
     pass
#

#%%artifact detection (normally during RAW stage but to save memory load in epoched)
"""
ssp_projectors = raw.info['projs']
raw.del_proj()
#visualize low frequency drifts
mag_channels = mne.pick_types(raw.info, meg='mag')
raw.plot(order=mag_channels, n_channels=len(mag_channels),
         remove_dc=False)
#visualize power line noise(spectrum)
fig = raw.plot_psd(tmax=np.inf, fmax=250, average=True)
    ## add some arrows at 60 Hz and its harmonics:
for ax in fig.axes[:2]:
    freqs = ax.lines[-1].get_xdata()
    psds = ax.lines[-1].get_ydata()
    for freq in (60, 120, 180, 240):
        idx = np.searchsorted(freqs, freq)
        ax.arrow(x=freqs[idx], y=psds[idx] + 18, dx=0, dy=-12, color='red',
                 width=0.1, head_width=3, length_includes_head=True)
#visuallize ECG
ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
ecg_epochs.plot_image(combine='mean')

avg_ecg_epochs = ecg_epochs.average()
avg_ecg_epochs.plot_topomap(times=np.linspace(-0.05, 0.05, 11))
'''