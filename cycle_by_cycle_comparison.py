
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
import seaborn as sns
import scipy as sp
from scipy import stats

import mne
from mne import combine_evoked
from mne.minimum_norm import apply_inverse
from mne.datasets.brainstorm import bst_auditory
from mne.io import read_raw_ctf
import matplotlib.pyplot as plt
from bycycle.filt import lowpass_filter, bandpass_filter
from bycycle.cyclepoints import _fzerorise, _fzerofall, find_extrema, find_zerox
from bycycle.features import compute_features
from bycycle.burst import plot_burst_detect_params



print(__doc__)

#%% prepare for reading data
#set working director
os.chdir('D:/school/2e master/case studies/2020CAED')#! be sure to change this to your working directory
#use precomputed steps
use_precomputed = True 

#get the data in directory/download if not in the directory 
data_path = bst_auditory.data_path('D:/school/2e master/case studies/mne_data')


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
#raw.plot(events, event_color = {1: 'b', 2: 'r'})

#epoch
tmin,tmax = -0.5, 0 #start & end of time window used to reject epoch 
reject = dict(mag=4e-12, eog=250e-6) #rejection parameters of peak to peak amplitude 
epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, picks=['meg', 'eog'],
                    baseline=(None, 0), reject=reject, preload=False, ##epochs overlapping 'bad' segments removed 
                    proj=True)
epochs.drop_bad()
epochs_standard = mne.concatenate_epochs([epochs['standard'][range(40)], #only 40 good epochs from each run
                                          epochs['standard'][182:222]]) #first epoch from 2nd run corresponds to 182
epochs_standard.load_data()  # Resampling to save memory.
epochs_standard.resample(200, npad='auto')
epochs_deviant = epochs['deviant'].load_data()
epochs_deviant.resample(200, npad='auto')
del epochs
evoked_std = epochs_standard.average()
evoked_dev = epochs_deviant.average()

epochs_deviant.plot_psd_topomap(ch_type='mag', normalize=True)
#%%epoch after cue

tmin,tmax = 0, 0.5 #start & end of time window used to reject epoch 
reject = dict(mag=4e-12, eog=250e-6) #rejection parameters of peak to peak amplitude
epochs = mne.Epochs(raw, events, event_dict, tmin, tmax, picks=['meg', 'eog'],
                    baseline=(0, 0), reject=reject, preload=False, ##epochs overlapping 'bad' segments removed 
                    proj=True)
epochs.drop_bad()
epochs_standard_AF = mne.concatenate_epochs([epochs['standard'][range(40)], #only 40 good epochs from each run
                                          epochs['standard'][182:222]]) #first epoch from 2nd run corresponds to 182
epochs_standard_AF.load_data()  # Resampling to save memory.
epochs_standard_AF.resample(200, npad='auto')
epochs_deviant_AF = epochs['deviant'].load_data()
epochs_deviant_AF.resample(200, npad='auto')
del epochs
# evoked_std_AF = epochs_standard_target.average()
# evoked_dev_AF = epochs_deviant_target.average()
epochs_deviant_AF.plot_psd_topomap(ch_type='mag', normalize=True)

#%%pick channels
Fc = [ 'MZF01-4408',
           'MZF02-4408',
           'MZF03-4408']
Fl = ['MLF11-4408',
         'MLF12-4408',
         'MLF13-4408',
         'MLF14-4408',
         'MLF21-4408',
         'MLF22-4408',
         'MLF23-4408',
         'MLF24-4408',
         'MLF25-4408',
         'MLF31-4408',
         'MLF32-4408',
         'MLF33-4408',
         'MLF34-4408',
         'MLF35-4408',
         'MLF41-4408',
         'MLF42-4408',
         'MLF43-4408',
         'MLF44-4408',
         'MLF45-4408',
         'MLF46-4408',
         'MLF51-4408',
         'MLF52-4408',
         'MLF53-4408',
         'MLF54-4408',
         'MLF55-4408',
         'MLF56-4408',
         'MLF61-4408',
         'MLF62-4408',
         'MLF63-4408',
         'MLF64-4408',
         'MLF65-4408',
         'MLF66-4408',
         'MLF67-4408'] 
Fr= [ 'MRF11-4408',
         'MRF12-4408',
         'MRF13-4408',
         'MRF14-4408',
         'MRF21-4408',
         'MRF22-4408',
         'MRF23-4408',
         'MRF24-4408',
         'MRF25-4408',
         'MRF31-4408',
         'MRF32-4408',
         'MRF33-4408',
         'MRF34-4408',
         'MRF35-4408',
         'MRF41-4408',
         'MRF42-4408',
         'MRF43-4408',
         'MRF44-4408',
         'MRF45-4408',
         'MRF46-4408',
         'MRF51-4408',
         'MRF52-4408',
         'MRF54-4408',
         'MRF55-4408',
         'MRF56-4408',
         'MRF61-4408',
         'MRF62-4408',
         'MRF63-4408',
         'MRF64-4408',
         'MRF65-4408',
         'MRF66-4408',
         'MRF67-4408',]
frontal= Fc+Fl+Fr



Oc = [ 'MZO01-4408',
 'MZO02-4408',
 'MZO03-4408']
Ol = [ 'MLO11-4408',
 'MLO12-4408',
 'MLO13-4408',
 'MLO14-4408',
 'MLO21-4408',
 'MLO22-4408',
 'MLO23-4408',
 'MLO24-4408',
 'MLO31-4408',
 'MLO32-4408',
 'MLO33-4408',
 'MLO34-4408',
 'MLO41-4408',
 'MLO44-4408',
 'MLO51-4408',
 'MLO53-4408']
Or =  ['MRO11-4408',     
 'MRO12-4408',
 'MRO13-4408',
 'MRO14-4408',
 'MRO21-4408',
 'MRO22-4408',
 'MRO23-4408',
 'MRO24-4408',
 'MRO31-4408',
 'MRO32-4408',
 'MRO33-4408',
 'MRO34-4408',
 'MRO41-4408',
 'MRO42-4408',
 'MRO43-4408',
 'MRO44-4408',
 'MRO51-4408',
 'MRO52-4408',
 'MRO53-4408']

occipital = Oc+Ol+Or
#%%start by cycle analysis
#preprocessing deviant/action
signal_frontal = []
for i in range(len(frontal)):
    signal = epochs_deviant.get_data(picks = frontal[i])#convert to numpy array 
    signal = signal[:,0,:]
    signal_frontal.append(np.ravel(signal))
del signal

signal_occipital = []
for i in range(len(occipital)):
    signal = epochs_deviant.get_data(picks = occipital[i])
    signal = signal[:,0,:]
    signal_occipital.append(np.ravel(signal))    
del signal

signals = [signal_frontal,signal_occipital]
signal = signals
Fs = 200
f_beta = (13, 30)
f_lowpass = 40
N_seconds = .2

#lowpass at 40 Hz
for i in range(len(signals)):
    for j in range(len(signals[i])):
        signals[i][j] = lowpass_filter(signals[i][j], Fs, f_lowpass,
                            N_seconds=N_seconds, remove_edge_artifacts=False)
        
#preprocessing after cue
signal_frontal_AF = []
for i in range(len(frontal)):
    signal = epochs_deviant_AF.get_data(picks = frontal[i])#convert to numpy array 
    signal = signal[:,0,:]
    signal_frontal_AF.append(np.ravel(signal))
del signal

signal_occipital_AF = []
for i in range(len(occipital)):
    signal = epochs_deviant_AF.get_data(picks = occipital[i])
    signal = signal[:,0,:]
    signal_occipital_AF.append(np.ravel(signal))    
del signal

signals_AF = [signal_frontal_AF,signal_occipital_AF]
signal = signals
Fs = 200
f_beta = (13, 30)
f_lowpass = 40
N_seconds = .2

#lowpass at 40 Hz
for i in range(len(signals_AF)):
    for j in range(len(signals_AF[i])):
        signals[i][j] = lowpass_filter(signals_AF[i][j], Fs, f_lowpass,
                            N_seconds=N_seconds, remove_edge_artifacts=False)
    
        
#%%
# Plot signal
N = len(signals[0])

T = len(signals[0][0])/Fs
t = np.arange(0, T, 1/Fs)


fig, axs = plt.subplots(3,figsize=(15, 9))
fig.suptitle('Frontal sensors')
axs[0].plot(t, signals[0][0], 'k')
axs[1].plot(t,signals[0][1],'k')
axs[2].plot(t,signals[0][2],'k')
#plt.xlim((0, T))
#plt.show()
fig, axs = plt.subplots(3,figsize=(15, 9))
fig.suptitle('Occipital sensors')
axs[0].plot(t, signals[1][0], 'k')
axs[1].plot(t,signals[1][1],'k')
axs[2].plot(t,signals[1][2],'k')

#localize peaks and troughts
Ps, Ts = find_extrema(signals[0][0], Fs, f_beta,
                      filter_kwargs={'N_seconds':N_seconds})
tlim = (12, 15)
tidx = np.logical_and(t>=tlim[0], t<tlim[1])
tidxPs = Ps[np.logical_and(Ps>tlim[0]*Fs, Ps<tlim[1]*Fs)]
tidxTs = Ts[np.logical_and(Ts>tlim[0]*Fs, Ts<tlim[1]*Fs)]

plt.figure(figsize=(12, 2))
plt.plot(t[tidx], signals[0][0][tidx], 'k')
plt.plot(t[tidxPs], signals[0][0][tidxPs], 'b.', ms=10)
plt.plot(t[tidxTs], signals[0][0][tidxTs], 'r.', ms=10)
plt.xlim(tlim)
plt.tight_layout()
plt.show()

# zero crossing & midpoints
from bycycle.cyclepoints import find_zerox
zeroxR, zeroxD = find_zerox(signals[0][0], Ps, Ts)
tlim = (12, 15)
tidx = np.logical_and(t>=tlim[0], t<tlim[1])
tidxPs = Ps[np.logical_and(Ps>tlim[0]*Fs, Ps<tlim[1]*Fs)]
tidxTs = Ts[np.logical_and(Ts>tlim[0]*Fs, Ts<tlim[1]*Fs)]
tidxDs = zeroxD[np.logical_and(zeroxD>tlim[0]*Fs, zeroxD<tlim[1]*Fs)]
tidxRs = zeroxR[np.logical_and(zeroxR>tlim[0]*Fs, zeroxR<tlim[1]*Fs)]

plt.figure(figsize=(12, 2))
plt.plot(t[tidx], signals[0][0][tidx], 'k')
plt.plot(t[tidxPs], signals[0][0][tidxPs], 'b.', ms=10)
plt.plot(t[tidxTs], signals[0][0][tidxTs], 'r.', ms=10)
plt.plot(t[tidxDs], signals[0][0][tidxDs], 'm.', ms=10)
plt.plot(t[tidxRs], signals[0][0][tidxRs], 'g.', ms=10)
plt.xlim(tlim)
plt.xlabel('Time (seconds)')
plt.tight_layout()
plt.show()

#%%cycle feature computation
#before cue
burst_kwargs = {'amplitude_fraction_threshold': 0.2,
                'amplitude_consistency_threshold': .4,
                'period_consistency_threshold': .5,
                'monotonicity_threshold': .8}

dfs = []
for i in range(len(signals)):
    for j in range(len(signals[i])):
        df = compute_features(signals[i][j], Fs, f_beta,
                          burst_detection_method='cycles',
                          burst_detection_kwargs=burst_kwargs)
        if i == 0:
            df['region'] = 'frontal'
            df['sensor'] = frontal[j]
        else:
            df['region'] = 'occipital'
            df['sensor'] = occipital[j]
        dfs.append(df)
df_cycles = pd.concat(dfs)
#after cue
dfs_AF = []
for i in range(len(signals_AF)):
    for j in range(len(signals_AF[i])):
        df_AF = compute_features(signals_AF[i][j], Fs, f_beta,
                          burst_detection_method='cycles',
                          burst_detection_kwargs=burst_kwargs)
        if i == 0:
            df_AF['region'] = 'frontal'
            df_AF['sensor'] = frontal[j]
        else:
            df_AF['region'] = 'occipital'
            df_AF['sensor'] = occipital[j]
        dfs_AF.append(df_AF)
df_cycles_AF = pd.concat(dfs_AF)


#%% confirm appropriateness of burst detection parameters
region = 'frontal'
sensor = 'MZF01-4408'
signal_df = df_cycles[df_cycles['sensor']==sensor]

if region == 'frontal':
    region = 0
    sensor = frontal.index(sensor)
else:
    region = 1
    sensor = occipital.index(sensor)
    
from bycycle.burst import plot_burst_detect_params
plot_burst_detect_params(signals[region][sensor], Fs, signal_df,
                         burst_kwargs, tlims=(0,35), figsize=(16, 3), plot_only_result=True)

plot_burst_detect_params(signals[region][sensor], Fs, signal_df,
                         burst_kwargs, tlims=(0,5), figsize=(16, 3))

#%%
# Only consider cycles that were identified to be in bursting regimes - Before cue
df_cycles_burst = df_cycles[df_cycles['is_burst']]

# Compute average features across sensors in a recording
features_keep = ['volt_amp', 'period', 'time_rdsym', 'time_ptsym']
df_sensor = df_cycles_burst.groupby(['region','sensor']).mean()[features_keep].reset_index()

df_sensor['volt_amp']=df_sensor['volt_amp']*10**13


feature_names = {'volt_amp': 'Amplitude (fT)',
                 'period': 'Period (ms)',
                 'time_rdsym': 'Rise-decay symmetry',
                 'time_ptsym': 'Peak-trough symmetry'}

for feat, feat_name in feature_names.items():
    g = sns.catplot(x='region', y=feat, data=df_sensor)
    plt.xlabel('')
    plt.xticks(size=20)
    plt.ylabel(feat_name, size=20)
    plt.yticks(size=15)
    plt.tight_layout()
    plt.show()

for feat, feat_name in feature_names.items():
    x_frontal = df_sensor[df_sensor['region']=='frontal'][feat]
    x_occipital = df_sensor[df_sensor['region']=='occipital'][feat]
    U, p = stats.mannwhitneyu(x_frontal, x_occipital)
    print('{:20s} difference between groups, U= {:3.0f}, p={:.5f}'.format(feat_name, U, p))
#%%
# Only consider cycles that were identified to be in bursting regimes - After cue
df_cycles_burst_AF = df_cycles_AF[df_cycles_AF['is_burst']]

# Compute average features across sensors in a recording
features_keep = ['volt_amp', 'period', 'time_rdsym', 'time_ptsym']
df_sensor_AF = df_cycles_burst_AF.groupby(['region','sensor']).mean()[features_keep].reset_index()

df_sensor_AF['volt_amp']=df_sensor_AF['volt_amp']*10**13


feature_names = {'volt_amp': 'Amplitude (fT)',
                 'period': 'Period (ms)',
                 'time_rdsym': 'Rise-decay symmetry',
                 'time_ptsym': 'Peak-trough symmetry'}

for feat, feat_name in feature_names.items():
    g = sns.catplot(x='region', y=feat, data=df_sensor_AF)
    plt.xlabel('')
    plt.xticks(size=20)
    plt.ylabel(feat_name, size=20)
    plt.yticks(size=15)
    plt.tight_layout()
    plt.show()

for feat, feat_name in feature_names.items():
    x_frontal = df_sensor_AF[df_sensor_AF['region']=='frontal'][feat]
    x_occipital = df_sensor_AF[df_sensor_AF['region']=='occipital'][feat]
    U, p = stats.mannwhitneyu(x_frontal, x_occipital)
    print('{:20s} difference between groups, U= {:3.0f}, p={:.5f}'.format(feat_name, U, p))
    
for feat, feat_name in feature_names.items():
    g = sns.catplot(x='region', y=feat, data=df_sensor_AF)
    plt.xlabel('')
    plt.xticks(size=20)
    plt.ylabel(feat_name, size=20)
    plt.yticks(size=15)
    plt.tight_layout()
    plt.show()

#%%
for feat, feat_name in feature_names.items():
    g = sns.catplot(x='region', y=feat, data=df_sensor_AF)
    plt.xlabel('')
    plt.xticks(size=20)
    plt.ylabel(feat_name, size=20)
    plt.yticks(size=15)
    plt.tight_layout()
    plt.show()

for feat, feat_name in feature_names.items():
    x_frontal = df_sensor[df_sensor['region']=='occipital'][feat]
    x_frontal_AF = df_sensor_AF[df_sensor_AF['region']=='occipital'][feat]
    U, p = stats.mannwhitneyu(x_frontal, x_frontal_AF)
    print('{:20s} difference between groups, U= {:3.0f}, p={:.5f}'.format(feat_name, U, p))
#%%dual amplitude threshold algorithm
    
from neurodsp import sim
from neurodsp.utils import create_times
from neurodsp.burst import detect_bursts_dual_threshold, compute_burst_stats
from neurodsp.plts.time_series import plot_time_series, plot_bursts
amp_dual_thresh = (1,1.9)  
f_range = (15, 30)
times = create_times(T, Fs)

bursting_df = []
for i in range(len(signal)):
    for j in range(len(signal[i])):
        bursting = detect_bursts_dual_threshold(signal[i][j], 200, amp_dual_thresh, f_range)        
        bursting_df.append(bursting)


sensors = frontal+occipital
stats_df= []
for i in range(len(bursting_df)):
    burst_stats=compute_burst_stats(bursting_df[i], Fs)
    burst_stats.update({'sensor':sensors[i]})
    if i > 67:
        burst_stats.update({'region':'occipital'})
    else: 
        burst_stats.update({'region':'frontal'})
    stats_df.append(burst_stats)
    
    
stats_df =pd.DataFrame.from_dict(stats_df)
#%%
features_keep = ['n_bursts', 'duration_mean']
df_sensorDual = stats_df.groupby(['region','sensor']).mean()[features_keep].reset_index()

feature_names = {'n_bursts': 'bursts',
                 'duration_mean': 'Period (ms)'}


for feat, feat_name in feature_names.items():
    g = sns.catplot(x='region', y=feat, data=df_sensorDual)
    plt.xlabel('')
    plt.xticks(size=20)
    plt.ylabel(feat_name, size=20)
    plt.yticks(size=15)
    plt.tight_layout()
    plt.show()
    
for feat, feat_name in feature_names.items():
    x_frontal = df_sensorDual[df_sensorDual['region']=='frontal'][feat]
    x_frontal = df_sensor[df_sensorDual['region']=='occipital'][feat]
    U, p = stats.mannwhitneyu(x_frontal, x_occipital)
    print('{:20s} difference between groups, U= {:3.0f}, p={:.5f}'.format(feat_name, U, p))

#%%
        
plot_bursts(times,signals[0][0],bursting_df[0],labels=['Raw Data', 'Detected Bursts'])
plot_instantaneous_measure(times,signals[0][0], measure = 'amplitude')
#%% preprocessing standard

signal_std_frontal = []
for i in range(len(frontal)):
    signal = epochs_standard.get_data(picks = frontal[i])#convert to numpy array 
    signal = signal[:,0,:]
    signal_std_frontal.append(np.ravel(signal))
del signal

signal_std_occipital = []
for i in range(len(occipital)):
    signal = epochs_standard.get_data(picks = occipital[i])
    signal = signal[:,0,:]
    signal_std_occipital.append(np.ravel(signal))    
del signal

signals_std = [signal_std_frontal,signal_std_occipital]
signal = signals
Fs = 200
f_beta = (13, 30)
f_lowpass = 40
N_seconds = .2

#lowpass at 200 Hz
for i in range(len(signals_std)):
    for j in range(len(signals_std[i])):
        signals_std[i][j] = lowpass_filter(signals_std[i][j], Fs, f_lowpass,
                            N_seconds=N_seconds, remove_edge_artifacts=False)
        
#%%cycle feature computation standard
burst_kwargs = {'amplitude_fraction_threshold': 0.2,
                'amplitude_consistency_threshold': .3,
                'period_consistency_threshold': .5,
                'monotonicity_threshold': .6,}

dfs = []
for i in range(len(signals_std)):
    for j in range(len(signals_std[i])):
        df = compute_features(signals_std[i][j], Fs, f_beta,
                          burst_detection_method='cycles',
                          burst_detection_kwargs=burst_kwargs)
        if i == 0:
            df['region'] = 'frontal'
            df['sensor'] = frontal[j]
        else:
            df['region'] = 'occipital'
            df['sensor'] = occipital[j]
        dfs.append(df)
df_std_cycles = pd.concat(dfs)
#%% confirm appropriateness of burst detection parameters standard
region = 'frontal'
sensor = 'MZF01-4408'
signal_std_df = df_std_cycles[df_std_cycles['sensor']==sensor]

if region == 'frontal':
    region = 0
    sensor = frontal.index(sensor)
else:
    region = 1
    sensor = occipital.index(sensor)
    
from bycycle.burst import plot_burst_detect_params
plot_burst_detect_params(signals_std[region][sensor], Fs, signal_std_df,
                         burst_kwargs, tlims=(0,10), figsize=(16, 3), plot_only_result=True)

plot_burst_detect_params(signals_std[region][sensor], Fs, signal_std_df,
                         burst_kwargs, tlims=(0,5), figsize=(16, 3))
#%%
# Only consider cycles that were identified to be in bursting regimes
df_std_cycles_burst = df_std_cycles[df_std_cycles['is_burst']]

# Compute average features across sensors in a recording
features_keep = ['volt_amp', 'period', 'time_rdsym', 'time_ptsym']
df_std_sensor = df_std_cycles_burst.groupby(['region','sensor']).mean()[features_keep].reset_index()

df_std_sensor['volt_amp']=df_std_sensor['volt_amp']*10**13


feature_names = {'volt_amp': 'Amplitude (fT)',
                 'period': 'Period (ms)',
                 'time_rdsym': 'Rise-decay symmetry',
                 'time_ptsym': 'Peak-trough symmetry'}

for feat, feat_name in feature_names.items():
    g = sns.catplot(x='region', y=feat, data=df_sensor)
    plt.xlabel('')
    plt.xticks(size=20)
    plt.ylabel(feat_name, size=20)
    plt.yticks(size=15)
    plt.tight_layout()
    plt.show()

for feat, feat_name in feature_names.items():
    x_frontal = df_std_sensor[df_std_sensor['region']=='frontal'][feat]
    x_occipital = df_std_sensor[df_std_sensor['region']=='occipital'][feat]
    U, p = stats.mannwhitneyu(x_frontal, x_occipital)
    print('{:20s} difference between groups, U= {:3.0f}, p={:.5f}'.format(feat_name, U, p))
    
for feat,feat_name in feature_names.items():
    plt
#%%
#epoch
tmin,tmax = 0, 0.5 #start & end of time window used to reject epoch 
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
