#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 23:08:14 2024

@author: supark
"""
#%%

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.stats import circmean
import os.path
from os import path
from scipy.io import savemat

# change working directory to where all my functions are
os.chdir('/mnt/neurocube/local/serenceslab/sunyoung/RNN/')
from circ_reg_fromJohn import *
from circ_corr import *

#%%

def fitCRF(x,y,init_params,doPlot):
    # Naka-Rushton equation from Sirawaj's paper
    # G_r: multiplicative response gain - since all CRFs asymptote at 80Hz, this should be constant
    # G_c: contrast gain factor that controls the horizontal shift of the CRF ** this is what we want to know
    # b: response baseline offset - can be fixed as the average of the minimum amplitude across all experimental conditions
    # q: exponent that controls the speed at which the CRF rises and reaches asymptote
    # Gr and Gc were constrained so that they could not be less than 0 and 1, respectively.
    # The exponent q was also constrained within a range of -10 to 10.
    # We used the 30% contrast value (about half of 61.66% contrast) as the initial seed value for Gc, 
    # the difference between maximum and minimum responses as the seed value for Gr, 
    # and 1 and 5 for the seed values of the exponent q when fitting the CRFs based on the P1 and the LPD (see below for LPD), respectively. 
    # The initial seed values for the exponent q were adopted from the estimated values based on a previous study
    # 0%, 2.24%, 5.13%, 11.75%, 26.92%, and 61.66%
    # R = G_r*c**q/(c**q+G_c**q)+b
    import scipy.optimize 
    
    # parameters for the equation
    b = np.min(y) # baseline offset
    G_r = np.max(y)-np.min(y) # multiplicative gain
    # init values arbitrarily chosen
    # init_q=5.5 # controls the steepness
    # init_G_c = 0.31 # horizontal shift
    (init_q,init_G_c) = init_params
    bnd_q=(-10,10)
    bnd_G_c=(-1,1)
    
    # define the equation
    def CRF(params,x):
        q=params[0]
        G_c=params[1]
        x = x.astype(float)
        R = G_r*(x**q)/(x**q+G_c**q)+b
        return R
    
    # a general use RSS function
    def rss_fun(fun):
        def loss_fun(params,x,y):
            return np.sqrt(np.mean((y-fun(params,x))**2))
        return loss_fun
    
    correction_fun = rss_fun(CRF)
    this_fit = scipy.optimize.minimize(correction_fun,(init_q,init_G_c),(x,y),bounds=(bnd_q,bnd_G_c))
    
    y_hat = CRF(this_fit.x,x)
    this_rss = np.mean((y-y_hat)**2)
    
    if doPlot == 1:
        plt.plot(y_hat)
        plt.plot(y)
        plt.title(f'RSS: {this_rss: .4}')
        plt.legend(['Fitted','Data'])
        plt.show()
        
    return this_fit.x, this_rss


def gridfitCRF(x,y,doPlot):
    # Naka-Rushton equation from Sirawaj's paper
    # G_r: multiplicative response gain - since all CRFs asymptote at 80Hz, this should be constant
    # G_c: contrast gain factor that controls the horizontal shift of the CRF ** this is what we want to know
    # b: response baseline offset - can be fixed as the average of the minimum amplitude across all experimental conditions
    # q: exponent that controls the speed at which the CRF rises and reaches asymptote
    # Gr and Gc were constrained so that they could not be less than 0 and 1, respectively.
    # The exponent q was also constrained within a range of -10 to 10.
    # We used the 30% contrast value (about half of 61.66% contrast) as the initial seed value for Gc, 
    # the difference between maximum and minimum responses as the seed value for Gr, 
    # and 1 and 5 for the seed values of the exponent q when fitting the CRFs based on the P1 and the LPD (see below for LPD), respectively. 
    # The initial seed values for the exponent q were adopted from the estimated values based on a previous study
    # 0%, 2.24%, 5.13%, 11.75%, 26.92%, and 61.66%
    # R = G_r*c**q/(c**q+G_c**q)+b
    # import scipy.optimize 
    
    # parameters for the equation
    b = np.min(y) # baseline offset
    G_r = np.max(y)-np.min(y) # multiplicative gain
    bnd_q=(-10,10) # controls the steepness
    bnd_G_c=(-1,1) # horizontal shift
    step_q=0.1
    step_G_c=0.01
    range_q = np.arange(bnd_q[0],bnd_q[1]+step_q,step_q)
    range_G_c = np.arange(bnd_G_c[0],bnd_G_c[1]+step_G_c,step_G_c)
    grid_rss = np.full((len(range_q), len(range_G_c)),np.nan)
    
    # define the equation
    def CRF(params,x):
        q=params[0]
        G_c=params[1]
        x = x.astype(float)
        R = G_r*(x**q)/(x**q+G_c**q)+b
        return R
    
    # a general use RSS function
    def rss_fun(fun):
        def loss_fun(params,x,y):
            return np.sqrt(np.mean((y-fun(params,x))**2))
        return loss_fun
    
    correction_fun = rss_fun(CRF)
    
    for cnt_q, q in enumerate(range_q):
        for cnt_G_c, G_c in enumerate(range_G_c):
            grid_rss[cnt_q,cnt_G_c] = correction_fun((q,G_c),x,y)
            
    # plt.imshow(grid_rss)
    # plt.colorbar()
    # plt.show()

    rss_flat = grid_rss.flatten()
    min_idx = np.nanargmin(rss_flat)
    q_idx = np.floor(min_idx/len(range_q)).astype(int)
    G_c_idx = np.mod(min_idx,len(range_G_c)).astype(int)
    # print(np.nanmin(rss_flat)==grid_rss[q_idx,G_c_idx])
    
    y_hat = CRF((range_q[q_idx], range_G_c[G_c_idx]),x)
    this_rss = np.mean((y-y_hat)**2)
    
    if doPlot == 1:
        plt.plot(y)
        plt.plot(y_hat)
        
        plt.title(f'RSS: {this_rss: .4}, G_c: {range_G_c[G_c_idx]: .2}')
        plt.legend(['Data','Fitted'])
        plt.show()
        
    return (range_q[q_idx], range_G_c[G_c_idx]), this_rss

def fitCRF2(x,y,doPlot):
    # Naka-Rushton equation from Sirawaj's paper
    # G_r: multiplicative response gain - since all CRFs asymptote at 80Hz, this should be constant
    # G_c: contrast gain factor that controls the horizontal shift of the CRF ** this is what we want to know
    # b: response baseline offset - can be fixed as the average of the minimum amplitude across all experimental conditions
    # q: exponent that controls the speed at which the CRF rises and reaches asymptote
    # Gr and Gc were constrained so that they could not be less than 0 and 1, respectively.
    # The exponent q was also constrained within a range of -10 to 10.
    # We used the 30% contrast value (about half of 61.66% contrast) as the initial seed value for Gc, 
    # the difference between maximum and minimum responses as the seed value for Gr, 
    # and 1 and 5 for the seed values of the exponent q when fitting the CRFs based on the P1 and the LPD (see below for LPD), respectively. 
    # The initial seed values for the exponent q were adopted from the estimated values based on a previous study
    # 0%, 2.24%, 5.13%, 11.75%, 26.92%, and 61.66%
    # R = G_r*c**q/(c**q+G_c**q)+b
    import scipy.optimize 
    
    # parameters for the equation
    b = np.min(y) # baseline offset
    G_r = np.max(y)-np.min(y) # multiplicative gain
    bnd_q=(-10,10) # controls the steepness
    bnd_G_c=(-1,1) # horizontal shift
    step_q=0.1
    step_G_c=0.01
    range_q = np.arange(bnd_q[0],bnd_q[1]+step_q,step_q)
    range_G_c = np.arange(bnd_G_c[0],bnd_G_c[1]+step_G_c,step_G_c)
    grid_rss = np.full((len(range_q), len(range_G_c)),np.nan)
    
    # define the equation
    def CRF(params,x):
        q=params[0]
        G_c=params[1]
        x = x.astype(float)
        R = G_r*(x**q)/(x**q+G_c**q)+b
        return R
    
    # a general use RSS function
    def rss_fun(fun):
        def loss_fun(params,x,y):
            return np.sqrt(np.mean((y-fun(params,x))**2))
        return loss_fun
    
    correction_fun = rss_fun(CRF)
    
    for cnt_q, q in enumerate(range_q):
        for cnt_G_c, G_c in enumerate(range_G_c):
            grid_rss[cnt_q,cnt_G_c] = correction_fun((q,G_c),x,y)
            
    # plt.imshow(grid_rss)
    # plt.colorbar()
    # plt.show()

    rss_flat = grid_rss.flatten()
    min_idx = np.nanargmin(rss_flat)
    q_idx = np.floor(min_idx/len(range_q)).astype(int)
    G_c_idx = np.mod(min_idx,len(range_G_c)).astype(int)
    # print(np.nanmin(rss_flat)==grid_rss[q_idx,G_c_idx])
    init_q=range_q[q_idx]
    init_G_c=range_G_c[G_c_idx]
    
    y_hat_grid = CRF((init_q, init_G_c),x)
    grid_rss = np.mean((y-y_hat_grid)**2)
    
    this_fit = scipy.optimize.minimize(correction_fun,(init_q,init_G_c),(x,y),bounds=(bnd_q,bnd_G_c))
    
    y_hat = CRF(this_fit.x,x)
    this_rss = np.mean((y-y_hat)**2)
    
    if doPlot == 1:
        plt.plot(y)
        plt.plot(y_hat_grid)
        plt.plot(y_hat)
        plt.title(f'grid: {grid_rss: .4}, RSS: {this_rss: .4}, G_c: {init_G_c: .2}, {this_fit.x[1]: .2}')
        plt.legend(['Data','grid fit','Fitted'])
        plt.show()
        
    return this_fit.x, this_rss

#%%
N_reps = 10
reps = np.arange(1,N_reps+1)
for rep_cnt, rep in enumerate(reps):
    fileName = 'results/S_seed-'+str(rep)+'.npz'
    
    
    dataFile = fileName
    data = np.load(dataFile)
    N_stim_loc=data['N_stim_loc']
    r_stim_amps=data['r_stim_amps']
    r_stim_ratios=data['r_stim_ratios']
    s_stim_amps=data['s_stim_amps']
    S_N_pools=data['S_N_pools']
    S_N_neuron=data['S_N_neuron']
    
    fr_att_abs=data['fr_att_abs']
    
    label_attPool_main = data['label_attPool_main']
    label_stim_strength_main = data['label_stim_strength_main']
    
    S_fr_avg_main = data['S_fr_avg_main']
    S_fr_avg_main_base = data['S_fr_avg_main_base']
    
    N_trials_attention = data['N_trials_attention']
    
    # combine data for higher attention gain sims
    fileName = 'results/S_higher-Rstim_seed-'+str(rep)+'.npz'    
    dataFile = fileName
    data = np.load(dataFile)    
    r_stim_amps = np.append(r_stim_amps,data['r_stim_amps'])
    S_fr_avg_main = np.concatenate((S_fr_avg_main,data['S_fr_avg_main']),axis=2)
    
    #% All plots in one
    mag_att = np.full((S_N_pools,len(s_stim_amps),2),np.nan)
    avg_FR = np.full((S_N_neuron,2,len(s_stim_amps)),np.nan)
    R_ratio = 0 # one r_stim_ratio for now
    
    # averaging FRs of neurons that prefer the stimulus
    N_stim_neuron = 2 #  94 by function
    Stim_idx = (np.arange(N_stim_neuron)+(S_N_neuron/2-N_stim_neuron/2)).astype(int)
    # avg_FR_stim = np.full((len(s_stim_amps),2),np.nan)
    avg_FR_stim_trial = np.full((N_trials_attention,len(s_stim_amps),2),np.nan)
    
    if rep_cnt == 0:
        avg_FR_stim = np.full((len(s_stim_amps),len(r_stim_amps),2,N_reps),np.nan)
        cont_gain = np.full((len(r_stim_amps),2,N_reps),np.nan)
        
        # avg_FR_stim_unatt = np.full((len(s_stim_amps),len(r_stim_amps),N_reps),np.nan)
        # cont_gain_unatt = np.full((len(r_stim_amps),N_reps),np.nan)
    
    colors = plt.cm.viridis(np.linspace(1,0,len(r_stim_amps)))
    
    for r_cnt, R_stim in enumerate(r_stim_amps):
        for ss_cnt, S_stim in enumerate(s_stim_amps):    
            for att_pool in range(2):
                trial_idx = (label_attPool_main==att_pool) & (label_stim_strength_main==S_stim) # average separately for trials for attention to stim vs unstim pool
                
                avg_FR[:,att_pool,ss_cnt] = np.mean(S_fr_avg_main[trial_idx,0:S_N_neuron,r_cnt,0],axis=0) # trial-average of FR
                # mag_att[:,ss_cnt,att_pool] = np.mean(np.mean(fr_att_abs[trial_idx,:,-5:,r_cnt,R_ratio],axis=0),axis=1) # average over all trials and then the last 5 timewindows
                
                # fr_att_abs_norm = np.mean(fr_att_abs[:,0,-5:,r_cnt,R_ratio],axis=1)/np.mean(S_fr_avg_main[:,0:S_N_neuron,r_cnt,0],axis=1)
                # mag_att[0,ss_cnt,att_pool] = np.mean(fr_att_abs_norm[trial_idx],axis=0)
                # mag_att[:,ss_cnt,att_pool] = np.mean(np.mean(fr_att_abs[trial_idx,:,-5:,r_cnt,R_ratio],axis=0),axis=1)
                
                avg_FR_stim[ss_cnt,r_cnt,att_pool,rep_cnt] = np.mean(np.mean(S_fr_avg_main[trial_idx,:,r_cnt,0][:,Stim_idx],axis=0),axis=0) # trial-average of FR
                
                # Sanity check: comparing 0 r_stim FR 
                avg_FR_stim_trial[:,ss_cnt,att_pool] = np.mean(S_fr_avg_main[trial_idx,:,r_cnt,0][:,Stim_idx],axis=1)
            
        
        # curve fitting
        params0, rss = fitCRF2(s_stim_amps/20,avg_FR_stim[:,r_cnt,0,rep_cnt],0)
        if np.isnan(rss):
            cont_gain[r_cnt,0,rep_cnt] = nan
        else:
            cont_gain[r_cnt,0,rep_cnt] = params0[1]
        
        params1, rss = fitCRF2(s_stim_amps/20,avg_FR_stim[:,r_cnt,1,rep_cnt],0)
        if np.isnan(rss):
            cont_gain[r_cnt,1,rep_cnt] = nan
        else:
            cont_gain[r_cnt,1,rep_cnt] = params1[1]
        
#% Save out
# newdic = {'avg_FR_stim':avg_FR_stim,'cont_gain':cont_gain}
# savemat('figure/result_S.mat',newdic)
# print('Saved figure/result_S.mat')



#%% Plot CRF - when the stimulated sub-network was attended vs a different sub-network was attended
avg_FR_stim_att = avg_FR_stim[:,:,0,:]
avg_FR_stim_unatt = avg_FR_stim[:,:,1,:]
repavg_FR_stim_att = np.mean(avg_FR_stim_att,axis=2)
se_FR_stim_att = np.std(avg_FR_stim_att,axis=2)/np.sqrt(N_reps)
repavg_FR_stim_unatt = np.mean(avg_FR_stim_unatt,axis=2)
se_FR_stim_unatt = np.std(avg_FR_stim_unatt,axis=2)/np.sqrt(N_reps)

# line1 = np.full((len(r_stim_amps_F)),np.nan)

# for r_cnt_S, R_stim_S in enumerate(r_stim_amps_S): # loop over spatial attention first
plt.figure(figsize=(8,5))
for r_cnt, R_stim in enumerate(r_stim_amps):
    plt.plot(repavg_FR_stim_att[:,r_cnt],color=colors[r_cnt,:],marker='o',linestyle='solid',fillstyle='none',label=str(R_stim))
    yerr = se_FR_stim_att[:,r_cnt]
    plt.fill_between(s_stim_amps, repavg_FR_stim_att[:,r_cnt] - yerr, repavg_FR_stim_att[:,r_cnt] + yerr, color=colors[r_cnt,:],alpha=0.3)
    
    # plt.plot(repavg_FR_stim_unatt[:,r_cnt],color=colors[r_cnt,:],marker='o',linestyle='dotted',fillstyle='none',label=str(R_stim))
    # yerr = se_FR_stim_unatt[:,r_cnt]
    # plt.fill_between(s_stim_amps, repavg_FR_stim_unatt[:,r_cnt] - yerr, repavg_FR_stim_unatt[:,r_cnt] + yerr, color=colors[r_cnt,:],alpha=0.3)
   
    # plt.fill_between(x, y - yerr, y + yerr, alpha=0.3)
    # for rep_cnt, rep in enumerate(reps):
    #     plt.plot(avg_FR_stim[:,r_cnt_F,r_cnt_S,rep_cnt],color=colors[r_cnt_F,:])
 
plt.legend()
plt.xticks(s_stim_amps,s_stim_amps)
# plt.title('Spatial Att. Gain : '+str(R_stim_S))
plt.xlabel('Stimulus Strength')
plt.ylabel('Avg FR for preferred neurons')
plt.show()
    

#%% Plot contrast gain
plt.figure(figsize=(8,5))
repavg_cont_gain = np.nanmean(cont_gain[:,0,:],axis=1)
se_cont_gain = np.std(cont_gain[:,0,:],axis=1)/np.sqrt(N_reps)
# for r_cnt_S, R_stim_S in enumerate(r_stim_amps_S):
plt.plot(r_stim_amps,repavg_cont_gain,color='black')
yerr = se_cont_gain
plt.fill_between(r_stim_amps, repavg_cont_gain - yerr, repavg_cont_gain + yerr, color='black',alpha=0.3)

repavg_cont_gain = np.nanmean(cont_gain[:,1,:],axis=1)
se_cont_gain = np.std(cont_gain[:,1,:],axis=1)/np.sqrt(N_reps)
# for r_cnt_S, R_stim_S in enumerate(r_stim_amps_S):
plt.plot(r_stim_amps,repavg_cont_gain,color='grey',linestyle='dotted')
yerr = se_cont_gain
plt.fill_between(r_stim_amps, repavg_cont_gain - yerr, repavg_cont_gain + yerr, color='grey',alpha=0.3)

    # for rep_cnt, rep in enumerate(reps):
    #     plt.plot(cont_gain[:,r_cnt_S,rep_cnt],color=colors[r_cnt_S,:])

plt.xlabel('Spatial Attention Strength')
plt.ylabel('Contrast Gain')
# plt.legend()
plt.show()
    

#%%
# fr_att_abs_norm = np.mean(fr_att_abs[:,0,-5:,r_cnt,R_ratio],axis=1)/np.mean(S_fr_avg_main[:,0:S_N_neuron,r_cnt,0],axis=1)
baseline_FR = np.full((len(r_stim_amps),2,len(s_stim_amps)),np.nan)
for r_cnt, R_stim in enumerate(r_stim_amps):
    for ss_cnt, S_stim in enumerate(s_stim_amps):    
        for att_pool in range(2):
            trial_idx = (label_attPool_main==att_pool) & (label_stim_strength_main==S_stim) # average separately for trials for attention to stim vs unstim pool
            
            baseline_FR[r_cnt,att_pool,ss_cnt] = np.mean(np.mean(S_fr_avg_main_base[trial_idx,0:S_N_neuron,r_cnt,0],axis=0),axis=0) # trial-average of FR
            