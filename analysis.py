# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:31:32 2020

@author: andre
"""
from statsmodels.multivariate.manova import MANOVA

df_sensor["cue"] = "precue"
df_sensor_AF["cue"] = "postcue"
df_comparison = pd.concat([df_sensor, df_sensor_AF])

df_comparison=df_comparison.loc[df_comparison['region'] == 'frontal']
maov = MANOVA.from_formula('cue ~  volt_amp + period + time_rdsym + time_ptsym', data=df_comparison)
print(maov.mv_test())


feature_names = {'volt_amp': 'Amplitude (fT)',
                 'period': 'Period (ms)',
                 'time_rdsym': 'Rise-decay symmetry',
                 'time_ptsym': 'Peak-trough symmetry'}

for feat, feat_name in feature_names.items():
    g = sns.catplot(x='cue', y=feat, data=df_comparison)
    plt.xlabel('')
    plt.xticks(size=20)
    plt.ylabel(feat_name, size=20)
    plt.yticks(size=15)
    plt.tight_layout()
    plt.show()

    
del df_comparison
#%% comparison dual threshold

df_sensorDual['method'] = 'dual amp'
df_sensorDual=df_sensorDual.rename(columns={'duration_mean': 'Period (ms)')
df_sensorDual['Period (ms)'] = df_sensorDual['Period (ms)']*
df_sensor['method'] = 'cycle'

df_comp_method=pd.concat([df_sensor, df_sensorDual)
#%%comaprison occipital & frontal
df_sensor["cue"] = "precue"
df_sensor_AF["cue"] = "postcue"
df_comparison = pd.concat([df_sensor, df_sensor_AF])

df_comparison=df_comparison.loc[df_comparison['cue'] == 'precue'] 

maov = MANOVA.from_formula('region ~  volt_amp + period + time_rdsym + time_ptsym', data=df_comparison)
print(maov.mv_test())

for feat, feat_name in feature_names.items():
    g = sns.catplot(x='region', y=feat, data=df_comparison)
    plt.xlabel('')
    plt.xticks(size=20)
    plt.ylabel(feat_name, size=20)
    plt.yticks(size=15)
    plt.tight_layout()
    plt.show()


del df_comparison
df_comparison = pd.concat([df_sensor, df_sensor_AF])
df_comparison=df_comparison.loc[df_comparison['cue'] == 'postcue']

maov = MANOVA.from_formula('region ~  volt_amp + period + time_rdsym + time_ptsym', data=df_comparison)
print(maov.mv_test())

for feat, feat_name in feature_names.items():
    g = sns.catplot(x='region', y=feat, data=df_comparison)
    plt.xlabel('')
    plt.xticks(size=20)
    plt.ylabel(feat_name, size=20)
    plt.yticks(size=15)
    plt.tight_layout()
    plt.show()

