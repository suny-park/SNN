#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 23:08:14 2024

@author: supark
"""
#%%
# change working directory to where all my functions are


import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.stats import circmean
import os.path
from os import path

os.chdir('/mnt/neurocube/local/serenceslab/sunyoung/RNN/')
from circ_reg_fromJohn import *
from circ_corr import *

#%%
def fitCRF(x,y,doPlot):
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
    init_q=5.5 # controls the steepness
    init_G_c = 0.31 # horizontal shift
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
        plt.plot(y_hat)
        plt.plot(y)
        plt.title(f'RSS: {this_rss: .4}')
        plt.legend(['Fitted','Data'])
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
# fileName = 'results/spatial_test_s0-20_r0-20_rr0.2.npz'
# fileName = 'results/spatial_test_s0-20_r0-2_rr0.2_skap0.25_test.npz'
# fileName = 'results/spatial_test_s0-16_r0-2_rr0.2_skap0.25_test.npz'
# fileName = 'results/FxS_'+str(rand_kappa)+'_seed-'+str(rep)+'.npz'
N_reps = 10
reps = np.arange(1,N_reps+1)
rand_kappa = 0
s_stim_amps = np.arange(20) # strength of bottom-up stimulus
for rep_cnt, rep in enumerate(reps):
    fileName = 'results/FxS_seed-'+str(rep)+'.npz'
    # fileName = 'results/F_seed-'+str(rep+1)+'.npz'
    
    
    dataFile = fileName
    data = np.load(dataFile)
    N_stim_loc=data['N_stim_loc']
    r_stim_amps_F=data['r_stim_amps_F']
    r_stim_amps_S=data['r_stim_amps_S']
    
    r_stim_ratios=data['r_stim_ratios']
    s_stim_amps=data['s_stim_amps']
    S_N_pools=data['S_N_pools']
    S_N_neuron=data['S_N_neuron']
    
    fr_att_abs=data['fr_att_abs']
    
    label_stim_main = data['label_stim_main']
    label_attPool_main = data['label_attPool_main']
    label_stim_strength_main = data['label_stim_strength_main']
    
    S_fr_avg_main = data['S_fr_avg_main']
    # S_fr_avg_main_base = data['S_fr_avg_main_base']
    
    N_trials_attention = data['N_trials_attention']
    
    #% All plots in one
    # mag_att = np.full((S_N_pools,len(s_stim_amps),2),np.nan)
    # avg_FR = np.full((S_N_neuron,2,len(s_stim_amps)),np.nan)
    R_ratio = 0 # one r_stim_ratio for now
    
    # averaging FRs of neurons that prefer the stimulus
    N_stim_neuron = 2 #  94 by function
    center_on_feature = 1 # 1 if attended, 0 if unattended stimulus
    # Stim_idx = (np.arange(N_stim_neuron)+(S_N_neuron/2-N_stim_neuron/2)).astype(int)
    if rep_cnt == 0:
        avg_FR_stim = np.full((len(s_stim_amps),len(r_stim_amps_F),len(r_stim_amps_S),N_reps),np.nan)
        cont_gain = np.full((len(r_stim_amps_F),len(r_stim_amps_S),N_reps),np.nan)
    # avg_FR_stim_trial = np.full((N_trials_attention,len(s_stim_amps),2),np.nan)
    colors = plt.cm.viridis(np.linspace(1,0,len(r_stim_amps_F)))
    # cmap = plt.get_cmap('viridis')

    for r_cnt_S, R_stim_S in enumerate(r_stim_amps_S): # loop over spatial attention first
        for r_cnt_F, R_stim_F in enumerate(r_stim_amps_F):
            for ss_cnt, S_stim in enumerate(s_stim_amps):    
                att_pool = 0
                # for att_pool in np.arange(1): # only for attended sub-network only for now
                    # trial_idx = label_stim_strength_main==S_stim
                trial_idx = np.argwhere((label_attPool_main==att_pool) & (label_stim_strength_main==S_stim)) # average separately for trials for attention to stim vs unstim pool
                # raw FR
                # avg_FR[:,att_pool,ss_cnt] = np.mean(S_fr_avg_main[trial_idx,0:S_N_neuron,r_cnt_F,r_cnt_S],axis=0) # trial-average of FR
                
                # mag_att[:,ss_cnt,att_pool] = np.mean(np.mean(fr_att_abs[trial_idx,:,-5:,r_cnt,R_ratio],axis=0),axis=1) # average over all trials and then the last 5 timewindows
                
                # fr_att_abs_norm = np.mean(fr_att_abs[:,0,-5:,r_cnt_F,r_cnt_S],axis=1)/np.mean(S_fr_avg_main[:,0:S_N_neuron,r_cnt_F,r_cnt_S],axis=1)
                # mag_att[0,ss_cnt,att_pool] = np.mean(fr_att_abs_norm[trial_idx],axis=0)
                # mag_att[:,ss_cnt,att_pool] = np.mean(np.mean(fr_att_abs[trial_idx,:,-5:,r_cnt,R_ratio],axis=0),axis=1)
                temp = np.full((len(trial_idx)),np.nan)
                for tidx,trial in enumerate(trial_idx):
                    if center_on_feature == 1:
                        feature_idx = label_stim_main[trial]
                    elif center_on_feature == 0:
                        feature_idx = np.mod(label_stim_main[trial]+S_N_neuron/2,S_N_neuron)
                    Stim_idx = (np.arange(N_stim_neuron)+(feature_idx-N_stim_neuron/2)).astype(int)
                    
                    temp[tidx] = np.mean(S_fr_avg_main[trial,Stim_idx,r_cnt_F,r_cnt_S],axis=0)
                avg_FR_stim[ss_cnt,r_cnt_F,r_cnt_S,rep_cnt] = np.mean(temp,axis=0) # trial-average of FR
            # params, rss = gridfitCRF(s_stim_amps/20,avg_FR_stim[:,r_cnt_F,r_cnt_S,rep_cnt],1)
            params, rss = fitCRF2(s_stim_amps/20,avg_FR_stim[:,r_cnt_F,r_cnt_S,rep_cnt],1)
            if np.isnan(rss):
                cont_gain[r_cnt_F,r_cnt_S,rep_cnt] = nan
            else:
                cont_gain[r_cnt_F,r_cnt_S,rep_cnt] = params[1]
                    
                    # Sanity check: comparing 0 r_stim FR 
                    # avg_FR_stim_trial[:,ss_cnt,att_pool] = np.mean(S_fr_avg_main[trial_idx,:,r_cnt,0][:,Stim_idx],axis=1)
                
                # print('R_stim:',R_stim,' S_stim:',S_stim)
                # print('Stim>Unstim trial #:',np.sum(avg_FR_stim_trial[:,ss_cnt,0]>avg_FR_stim_trial[:,ss_cnt,1]),'/',N_trials_attention)
                    
            # mag_att2 = np.full((2,len(s_stim_amps)),np.nan)
            # mag_att2[0,:] = mag_att[0,:,0] # att == stim pool
            # mag_att2[1,:] = mag_att[0,:,1] # att != stim pool
            
            # if R_stim==0:
            #     fig0, axs = plt.subplots(2,10,figsize=(12,4))
            #     for ss_cnt, S_stim in enumerate(s_stim_amps):
            #         if ss_cnt<10:
            #             axs[0,ss_cnt].plot(avg_FR[:,:,ss_cnt])
            #             # ax1.legend(['Stimulated','Unstimulated'],frameon=False)
            #             axs[0,ss_cnt].set_title('S_stim = '+str(s_stim_amps[ss_cnt]))
            #         else:
            #             axs[1,ss_cnt-10].plot(avg_FR[:,:,ss_cnt])
            #             # ax1.legend(['Stimulated','Unstimulated'],frameon=False)
            #             axs[1,ss_cnt-10].set_title('S_stim = '+str(s_stim_amps[ss_cnt]))
        # plt.set_cmap(cmap)
        
#%%
repavg_FR_stim = np.mean(avg_FR_stim,axis=3)
for r_cnt_S, R_stim_S in enumerate(r_stim_amps_S): # loop over spatial attention first
    for r_cnt_F, R_stim_F in enumerate(r_stim_amps_F):
        plt.plot(repavg_FR_stim[:,r_cnt_F,r_cnt_S],color=colors[r_cnt_F,:])
 
    plt.legend(r_stim_amps_F)
    plt.title('Spatial Att. Gain : '+str(R_stim_S))
    plt.xlabel('Stimulus Strength')
    plt.ylabel('Avg FR for preferred neurons')
    plt.show()
#%%
repavg_cont_gain = np.nanmean(cont_gain,axis=2)
for r_cnt_S, R_stim_S in enumerate(r_stim_amps_S):
    plt.plot(repavg_cont_gain[:,r_cnt_S],color=colors[r_cnt_S,:])

plt.xlabel('Feature Attention Strength')
plt.ylabel('Contrast Gain')
plt.legend(r_stim_amps_S)
plt.show()
            
        # fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
        
        # ss_cnt = 1
        # ax1.plot(avg_FR[:,:,ss_cnt])
        # ax1.legend(['Stimulated','Unstimulated'],frameon=False)
        # ax1.set_title('S_stim = '+str(s_stim_amps[ss_cnt]))
        # # plt.show()
        # # print(np.mean(avg_FR[:,0,ss_cnt],axis=0), np.mean(avg_FR[:,1,ss_cnt],axis=0))
    
        # # plt.figure(figsize=(4,4))
        # ax2.plot(avg_FR_stim)
        # # plt.plot(np.mean(MAE2[:,0,:],axis=1),c='k',linewidth=2)
        # ax2.set_xlabel('Stimulus Strength')
        # ax2.set_ylabel('Avg FR for preferred neurons')
        # ax2.set_xticks(ticks=np.arange(len(s_stim_amps)),labels=s_stim_amps)
        # # plt.yticks(ticks=np.arange(5))
        # ax2.legend(['Stimulated','Unstimulated'],frameon=False)
        # # ax3.set_title('R_stim = '+str(r_stim_amps[r_cnt]))
        
        # plt.suptitle('R_stim = '+str(r_stim_amps[r_cnt]))
        # plt.tight_layout()
        # plt.show()
        

#%%
# fileName = 'results/spatial_test_s0-9_r7.npz'

# dataFile = fileName
# data = np.load(dataFile)
# N_stim_loc=data['N_stim_loc']
# r_stim_amps=data['r_stim_amps']
# r_stim_ratios=data['r_stim_ratios']
# s_stim_amps=data['s_stim_amps']
# S_N_pools=data['S_N_pools']
# S_N_neuron=data['S_N_neuron']

# fr_att_abs=data['fr_att_abs']

# label_attPool_main = data['label_attPool_main']

# mag_att = np.full((S_N_pools,len(r_stim_amps),2),np.nan)
# R_ratio = 0 # one r_stim_ratio for now
# for r_cnt, R_stim in enumerate(r_stim_amps):    
#     for att_pool in range(2):
#         trial_idx = label_attPool_main==att_pool # average separately for trials for attention to stim vs unstim pool
    
#         mag_att[:,r_cnt,att_pool] = np.mean(np.mean(fr_att_abs[trial_idx,:,-5:,r_cnt,R_ratio],axis=0),axis=1) # average over all trials and then the last 5 timewindows
    
# mag_att2 = np.full((2,len(r_stim_amps)),np.nan)
# mag_att2[0,:] = mag_att[0,:,0] # att == stim pool
# mag_att2[1,:] = mag_att[0,:,1] # att != stim pool


# plt.figure(figsize=(4,4))
# plt.plot(mag_att2[0,:],c='k',linewidth=2)
# plt.plot(mag_att2[1,:],'--k',linewidth=2)
# # plt.plot(np.mean(MAE2[:,0,:],axis=1),c='k',linewidth=2)
# plt.xlabel('Gain Strength to Second Layer')
# plt.ylabel('Likelihood magnitude at attended feature')
# plt.xticks(ticks=np.arange(len(r_stim_amps)),labels=r_stim_amps)
# plt.yticks(ticks=np.arange(5))
# plt.legend(['Stimulated','Unstimulated'],frameon=False)
# plt.title('kappa = '+str(rand_kappa))
# plt.show()



#%%

# np.mean(S_fr_avg_main[:,0:S_N_neuron,r_cnt,0],axis=1)
# fr_att_abs_norm = np.mean(fr_att_abs[:,0,-5:,r_cnt,R_ratio],axis=1)/np.mean(S_fr_avg_main[:,0:S_N_neuron,r_cnt,0],axis=1)
mag_att = np.full((S_N_pools,len(s_stim_amps),2),np.nan)
avg_FR = np.full((S_N_neuron,2,len(s_stim_amps)),np.nan)
R_ratio = 0 # one r_stim_ratio for now
for r_cnt, R_stim in enumerate(r_stim_amps):
    for ss_cnt, S_stim in enumerate(s_stim_amps):    
        for att_pool in range(2):
            trial_idx = (label_attPool_main==att_pool) & (label_stim_strength_main==S_stim) # average separately for trials for attention to stim vs unstim pool
            
            avg_FR[:,att_pool,ss_cnt] = np.mean(S_fr_avg_main[trial_idx,0:S_N_neuron,r_cnt,0],axis=0) # trial-average of FR
            # mag_att[:,ss_cnt,att_pool] = np.mean(np.mean(fr_att_abs[trial_idx,:,-5:,r_cnt,R_ratio],axis=0),axis=1) # average over all trials and then the last 5 timewindows
            
            fr_att_abs_norm = np.mean(fr_att_abs[:,0,-5:,r_cnt,R_ratio],axis=1)/np.mean(S_fr_avg_main[:,0:S_N_neuron,r_cnt,0],axis=1)
            mag_att[0,ss_cnt,att_pool] = np.mean(fr_att_abs_norm[trial_idx],axis=0)
            # mag_att[:,ss_cnt,att_pool] = np.mean(np.mean(fr_att_abs[trial_idx,:,-5:,r_cnt,R_ratio],axis=0),axis=1)
    mag_att2 = np.full((2,len(s_stim_amps)),np.nan)
    mag_att2[0,:] = mag_att[0,:,0] # att == stim pool
    mag_att2[1,:] = mag_att[0,:,1] # att != stim pool
    
    ss_cnt = 3
    plt.figure(figsize=(4,4))
    plt.plot(avg_FR[:,:,ss_cnt])
    plt.title('R_stim = '+str(r_stim_amps[r_cnt]))
    plt.show()
    # print(np.mean(avg_FR[:,0,ss_cnt],axis=0), np.mean(avg_FR[:,1,ss_cnt],axis=0))

    
    plt.figure(figsize=(4,4))
    plt.plot(mag_att2[0,:],c='k',linewidth=2)
    plt.plot(mag_att2[1,:],'--k',linewidth=2)
    # plt.plot(np.mean(MAE2[:,0,:],axis=1),c='k',linewidth=2)
    plt.xlabel('Stimulus Strength')
    plt.ylabel('Likelihood magnitude at presented feature')
    plt.xticks(ticks=np.arange(len(s_stim_amps)),labels=s_stim_amps)
    # plt.yticks(ticks=np.arange(5))
    plt.legend(['Stimulated','Unstimulated'],frameon=False)
    plt.title('R_stim = '+str(r_stim_amps[r_cnt]))
    plt.show()

#%%
# fr_att_abs_norm = np.mean(fr_att_abs[:,0,-5:,r_cnt,R_ratio],axis=1)/np.mean(S_fr_avg_main[:,0:S_N_neuron,r_cnt,0],axis=1)
baseline_FR = np.full((len(r_stim_amps),2,len(s_stim_amps)),np.nan)
for r_cnt, R_stim in enumerate(r_stim_amps):
    for ss_cnt, S_stim in enumerate(s_stim_amps):    
        for att_pool in range(2):
            trial_idx = (label_attPool_main==att_pool) & (label_stim_strength_main==S_stim) # average separately for trials for attention to stim vs unstim pool
            
            # avg_FR[:,att_pool,ss_cnt] = np.mean(S_fr_avg_main[trial_idx,0:S_N_neuron,r_cnt,0],axis=0) # trial-average of FR
            baseline_FR[r_cnt,att_pool,ss_cnt] = np.mean(np.mean(S_fr_avg_main_base[trial_idx,0:S_N_neuron,r_cnt,0],axis=0),axis=0) # trial-average of FR
            # mag_att[:,ss_cnt,att_pool] = np.mean(np.mean(fr_att_abs[trial_idx,:,-5:,r_cnt,R_ratio],axis=0),axis=1) # average over all trials and then the last 5 timewindows
            
            # fr_att_abs_norm = np.mean(fr_att_abs[:,0,-5:,r_cnt,R_ratio],axis=1)/np.mean(S_fr_avg_main[:,0:S_N_neuron,r_cnt,0],axis=1)
            # mag_att[0,ss_cnt,att_pool] = np.mean(fr_att_abs_norm[trial_idx],axis=0)
            # mag_att[:,ss_cnt,att_pool] = np.mean(np.mean(fr_att_abs[trial_idx,:,-5:,r_cnt,R_ratio],axis=0),axis=1)
    # mag_att2 = np.full((2,len(s_stim_amps)),np.nan)
    # mag_att2[0,:] = mag_att[0,:,0] # att == stim pool
    # mag_att2[1,:] = mag_att[0,:,1] # att != stim pool
    #%%
# for ss_cnt, S_stim in enumerate(s_stim_amps):    
plt.figure(figsize=(4,4))
plt.plot(np.mean(baseline_FR,axis=2))
plt.title('Pre-stimulus baseline FR change with top-down gain')
plt.legend(['Stimulated','Unstimulated'],frameon=False)
plt.xlabel('Top-down Gain Strength')
plt.ylabel('Average FR (Hz)')
# plt.ylim([0.17,0.24])
plt.show()
    # print(np.mean(avg_FR[:,0,ss_cnt],axis=0), np.mean(avg_FR[:,1,ss_cnt],axis=0))

#%% Average FR (instead of vector magnitude)
# 94 neurons are stimulaed from the stimulus
# average FR of 94 neurons and use that for the CRF
N_stim_neuron = 94
Stim_idx = (np.arange(N_stim_neuron)+(S_N_neuron/2-N_stim_neuron/2)).astype(int)
avg_FR = np.full((len(s_stim_amps),2),np.nan)
for r_cnt, R_stim in enumerate(r_stim_amps):
    for ss_cnt, S_stim in enumerate(s_stim_amps):    
        for att_pool in range(2):
            trial_idx = (label_attPool_main==att_pool) & (label_stim_strength_main==S_stim) # average separately for trials for attention to stim vs unstim pool

            avg_FR[ss_cnt,att_pool] = np.mean(np.mean(S_fr_avg_main[trial_idx,:,r_cnt,0][:,Stim_idx],axis=0),axis=0) # trial-average of FR
    plt.figure(figsize=(4,4))
    plt.plot(avg_FR)
    # plt.plot(np.mean(MAE2[:,0,:],axis=1),c='k',linewidth=2)
    plt.xlabel('Stimulus Strength')
    plt.ylabel('Avg FR for preferred neurons')
    plt.xticks(ticks=np.arange(len(s_stim_amps)),labels=s_stim_amps)
    # plt.yticks(ticks=np.arange(5))
    plt.legend(['Stimulated','Unstimulated'],frameon=False)
    plt.title('R_stim = '+str(r_stim_amps[r_cnt]))
    plt.show()
            
            
            
            