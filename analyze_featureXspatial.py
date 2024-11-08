#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 23:08:14 2024

@author: supark
"""
#%%
# change working directory to where all my functions are

import os.path
from os import path

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.stats import circmean
import seaborn as sns

from scipy.io import savemat

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
    # Naka-Rushton equation from Sirawaj's paper / code bits from Tim0
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

#%% CRF
N_reps = 10
reps = np.arange(1,N_reps+1)
# kappas = [0.1,0.2,0.3,0.4]
kappas = [0.4]
s_stim_amps = np.arange(20) # strength of bottom-up stimulus
for rand_kappa in kappas:
    for rep_cnt, rep in enumerate(reps):
        # fileName = 'results/FxS_kappa-'+str(rand_kappa)+'_seed-'+str(rep)+'.npz'
        fileName = 'results/FxS_seed-'+str(rep)+'.npz'
        # fileName = 'results/F_test_inh-0.9_seed-'+str(rep+1)+'.npz'
        
        
        dataFile = fileName
        data = np.load(dataFile)
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
        
        N_trials_attention = data['N_trials_attention']
        
        
        R_ratio = 0 # one r_stim_ratio for now
        
        # averaging FRs of neurons that prefer the stimulus
        N_stim_neuron = 2 #  94 by function
        if rep_cnt == 0:
            avg_FR_stim_att = np.full((len(s_stim_amps),len(r_stim_amps_F),len(r_stim_amps_S),N_reps),np.nan)
            cont_gain_att = np.full((len(r_stim_amps_F),len(r_stim_amps_S),N_reps),np.nan)
            
            avg_FR_stim_unatt = np.full((len(s_stim_amps),len(r_stim_amps_F),len(r_stim_amps_S),N_reps),np.nan)
            cont_gain_unatt = np.full((len(r_stim_amps_F),len(r_stim_amps_S),N_reps),np.nan)
        
        colors = plt.cm.viridis(np.linspace(1,0,len(r_stim_amps_F)))
    
        for r_cnt_S, R_stim_S in enumerate(r_stim_amps_S): # loop over spatial attention first
            for r_cnt_F, R_stim_F in enumerate(r_stim_amps_F):
                for ss_cnt, S_stim in enumerate(s_stim_amps):    
                    att_pool = 0
                    # for att_pool in np.arange(1): # only for attended sub-network only for now
                        # trial_idx = label_stim_strength_main==S_stim
                    trial_idx = np.argwhere((label_attPool_main==att_pool) & (label_stim_strength_main==S_stim)) # average separately for trials for attention to stim vs unstim pool
                    
                    # attended stimulus
                    temp = np.full((len(trial_idx)),np.nan)
                    for tidx,trial in enumerate(trial_idx):
                        feature_idx = label_stim_main[trial]
                        Stim_idx = (np.arange(N_stim_neuron)+(feature_idx-N_stim_neuron/2)).astype(int)
                        
                        temp[tidx] = np.mean(S_fr_avg_main[trial,Stim_idx,r_cnt_F,r_cnt_S],axis=0)
                    avg_FR_stim_att[ss_cnt,r_cnt_F,r_cnt_S,rep_cnt] = np.mean(temp,axis=0) # trial-average of FR
                    
                    # unattended stimulus
                    temp = np.full((len(trial_idx)),np.nan)
                    for tidx,trial in enumerate(trial_idx):
                        feature_idx = np.mod(label_stim_main[trial]+S_N_neuron/2,S_N_neuron)
                        Stim_idx = (np.arange(N_stim_neuron)+(feature_idx-N_stim_neuron/2)).astype(int)
                        
                        temp[tidx] = np.mean(S_fr_avg_main[trial,Stim_idx,r_cnt_F,r_cnt_S],axis=0)
                    avg_FR_stim_unatt[ss_cnt,r_cnt_F,r_cnt_S,rep_cnt] = np.mean(temp,axis=0) # trial-average of FR
                    
                # curve fitting
                params, rss = fitCRF2(s_stim_amps/20,avg_FR_stim_att[:,r_cnt_F,r_cnt_S,rep_cnt],0)
                if np.isnan(rss):
                    cont_gain_att[r_cnt_F,r_cnt_S,rep_cnt] = nan
                else:
                    cont_gain_att[r_cnt_F,r_cnt_S,rep_cnt] = params[1]
                    
                params, rss = fitCRF2(s_stim_amps/20,avg_FR_stim_unatt[:,r_cnt_F,r_cnt_S,rep_cnt],0)
                if np.isnan(rss):
                    cont_gain_unatt[r_cnt_F,r_cnt_S,rep_cnt] = nan
                else:
                    cont_gain_unatt[r_cnt_F,r_cnt_S,rep_cnt] = params[1]
    
    #% Save out CRF and cont gain
    
    # save kappa file separately
    # os.chdir('/mnt/neurocube/local/serenceslab/sunyoung/RNN/figure')
    
    datdic = {'avg_FR_stim_att':avg_FR_stim_att,'avg_FR_stim_unatt':avg_FR_stim_unatt,\
              'cont_gain_att':cont_gain_att,'cont_gain_unatt':cont_gain_unatt}
    savemat('figure/result_FxS_kappa-'+str(rand_kappa)+'.mat',datdic)
    print('Saved figure/result_FxS_kappa-'+str(rand_kappa)+'.mat')
#%% CRF - unstimulated sub-network
N_reps = 1 # only one reps for FxS kappa modulations (except for k=0)
kappas = [0,0.1,0.2,0.3,0.4]
for kidx,rand_kappa in enumerate(kappas):
    if rand_kappa == 0:
        dataFile = 'results/FxS_seed-1.npz'
    else:
        dataFile = 'results/FxS_kappa-'+str(rand_kappa)+'_seed-1.npz'
    
    data = np.load(dataFile)
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
    
    N_trials_attention = data['N_trials_attention']
    
    R_ratio = 0 # one r_stim_ratio for now
    
    # averaging FRs of neurons that prefer the stimulus
    N_stim_neuron = 2 #  94 by function
        
    avg_FR_stim = np.full((len(s_stim_amps),len(r_stim_amps_F),len(r_stim_amps_S),S_N_pools),np.nan)
    
    if kidx == 0: 
        avg_FR_stim_kappa = np.full((len(s_stim_amps),len(r_stim_amps_F),len(r_stim_amps_S),S_N_pools,len(kappas)),np.nan)
        # avg_FR_stim_att = np.full((len(s_stim_amps),len(r_stim_amps_F),len(r_stim_amps_S),len(kappas)),np.nan)
        # cont_gain_att = np.full((len(r_stim_amps_F),len(r_stim_amps_S),len(kappas)),np.nan)
        
        # avg_FR_stim_unatt = np.full((len(s_stim_amps),len(r_stim_amps_F),len(r_stim_amps_S),len(kappas)),np.nan)
        # cont_gain_unatt = np.full((len(r_stim_amps_F),len(r_stim_amps_S),len(kappas)),np.nan)
    
    # colors = plt.cm.viridis(np.linspace(1,0,len(r_stim_amps_F)))

    for r_cnt_S, R_stim_S in enumerate(r_stim_amps_S): # loop over spatial attention first
        for r_cnt_F, R_stim_F in enumerate(r_stim_amps_F):
            for ss_cnt, S_stim in enumerate(s_stim_amps):    
                att_pool = 0
                trial_idx = np.argwhere((label_attPool_main==att_pool) & (label_stim_strength_main==S_stim)) # average separately for trials for attention to stim vs unstim pool
                # doing this the long way... so that it works with whatever stim label
                for pool in np.arange(S_N_pools):
                    temp = np.full((len(trial_idx)),np.nan) # store average response for each trial (because stim center changes based on the trial)
                    for tidx, trial in enumerate(trial_idx):
                        feature_idx = label_stim_main[trial]
                        Stim_idx = (np.arange(N_stim_neuron)+(feature_idx-N_stim_neuron/2)+pool*S_N_neuron).astype(int)
                        
                        temp[tidx] = np.mean(S_fr_avg_main[trial,Stim_idx,r_cnt_F,r_cnt_S],axis=0)
                    avg_FR_stim[ss_cnt,r_cnt_F,r_cnt_S,pool] = np.mean(temp)
            # params, rss = fitCRF2(s_stim_amps/20,avg_FR_stim[:,r_cnt_F,r_cnt_S,0],0)
            
    avg_FR_stim_kappa[:,:,:,:,kidx] = avg_FR_stim
    # avg_FR_stim_att[:,:,:,kidx] = avg_FR_stim[:,:,:,0] # trial-average of FR
    # avg_FR_stim_unatt[:,:,:,kidx] = np.mean(avg_FR_stim[:,:,:,1:],axis=3) # trial-average of FR
    
    # curve fitting
    # for r_cnt_S, R_stim_S in enumerate(r_stim_amps_S): # loop over spatial attention first
    #     for r_cnt_F, R_stim_F in enumerate(r_stim_amps_F):
            
    #         params, rss = fitCRF2(s_stim_amps/20,avg_FR_stim_att[:,r_cnt_F,r_cnt_S,kidx],0)
    #         if np.isnan(rss):
    #             cont_gain_att[r_cnt_F,r_cnt_S,kidx] = nan
    #         else:
    #             cont_gain_att[r_cnt_F,r_cnt_S,kidx] = params[1]
                
    #         params, rss = fitCRF2(s_stim_amps/20,avg_FR_stim_unatt[:,r_cnt_F,r_cnt_S,kidx],0)
    #         if np.isnan(rss):
    #             cont_gain_unatt[r_cnt_F,r_cnt_S,kidx] = nan
    #         else:
    #             cont_gain_unatt[r_cnt_F,r_cnt_S,kidx] = params[1]
    
#% Save out CRF and cont gain
datdic = {'avg_FR_stim_kappa':avg_FR_stim_kappa}
savemat('figure/result_FxS_unstimPool.mat',datdic)
print('Saved figure/result_FxS_unstimPool.mat')

#%% MAE Analysis - HERE!!
# Train model with localizer task from feature attention sims to predict attended stimulus
# Test on main task data
rep=1

N_reps = 1 # only one reps for FxS kappa modulations (except for k=0)
kappas = [0,0.1,0.2,0.3,0.4]
for kidx,rand_kappa in enumerate(kappas):

    # Load F sensory task data
    dataFile = 'results/F_kappa-'+str(rand_kappa)+'_seed-'+str(rep)+'.npz'
    print('Loading '+dataFile)
    data = np.load(dataFile)
    
    # load variables
    S_fr_avg_loc=data['S_fr_avg_loc']
    label_stim_loc=data['label_stim_loc']
    label_pool_loc=data['label_pool_loc']
    label_trial_loc=data['label_trial_loc']
    
    N_stim_loc=data['N_stim_loc']
    
    S_N_pools=data['S_N_pools']
    S_N_neuron=data['S_N_neuron']
    
    # Load FxS attention task data
    if rand_kappa == 0:
        dataFile = 'results/FxS_seed-1.npz'
    else:
        dataFile = 'results/FxS_kappa-'+str(rand_kappa)+'_seed-1.npz'
    print('Loading '+dataFile)
    data = np.load(dataFile)
    
    r_stim_amps_F=data['r_stim_amps_F']
    r_stim_amps_S=data['r_stim_amps_S']
    
    r_stim_ratios=data['r_stim_ratios']
    s_stim_amps=data['s_stim_amps']
    S_N_pools=data['S_N_pools']
    S_N_neuron=data['S_N_neuron']
    
    label_stim_main = data['label_stim_main']
    label_attPool_main = data['label_attPool_main']
    label_stim_strength_main = data['label_stim_strength_main']
    
    S_fr_avg_main = data['S_fr_avg_main']
    
    N_trials_attention = data['N_trials_attention']
    
    
    # convert label (0-511) to degrees (0-2pi, 0-359)
    label_stim_main_deg = (label_stim_main/S_N_neuron*360).astype(int)
    label_stim_main_bin = np.full(len(label_stim_main),np.nan)
    for lmi,lm in enumerate(np.unique(label_stim_main)):
        label_stim_main_bin[label_stim_main==lm]=lmi
    label_stim_main_bin = label_stim_main_bin.astype(int)
    R_ratio = 0 # one r_stim_ratio for now
    
    label_stim_loc_deg = (label_stim_loc/S_N_neuron*360).astype(int)
    
    if kidx == 0:
        # np.full((len(s_stim_amps),len(r_stim_amps_F),len(r_stim_amps_S),S_N_pools),np.nan)
        decoding2 = np.full((len(label_trial_main),S_N_pools,2,len(r_stim_amps_F),len(r_stim_amps_S),len(kappas)),np.nan)
        MAE2 = np.full((len(s_stim_amps),len(r_stim_amps_F),len(r_stim_amps_S),S_N_pools,len(kappas)),np.nan)
        

# sig2 = np.full((S_N_pools,2),np.nan) # to bucket circ corr coef and p_vals
    for thispool in np.arange(S_N_pools):
        pool_idx = np.arange(thispool*S_N_neuron,(thispool+1)*S_N_neuron) # indices of neurons in thispool
        
        poolX = S_fr_avg_loc[label_pool_loc==thispool,:] # subset trials when this pool was stimulated
        trnX = poolX[:,pool_idx] # subset neurons within this pool
        trny = label_stim_loc_deg[label_pool_loc==thispool] # subset trials when this pool was stimulated    
        
        for r_cnt_F in np.arange(len(r_stim_amps_F)):
            for r_cnt_S in np.arange(len(r_stim_amps_S)):
                tstX = S_fr_avg_main[:,pool_idx,r_cnt_F,r_cnt_S] # subset neurons within this pool
                tsty = label_stim_main_deg
                
                my_model = compute_reg_weights(trnX,trny) # Circular regression with empirical bayes ridge
                pred = decode_reg(my_model,tstX)
                
                pred_deg = pred/np.pi*180 # convert radians back to degrees
                
                # store test & predicted labels and store them for visualization later
                # pred_deg[np.abs(tsty-pred_deg)>180] = np.abs(pred_deg[np.abs(tsty-pred_deg)>180]-360)
                decoding2[:,thispool,0,r_cnt_F,r_cnt_S,kidx] = tsty
                decoding2[:,thispool,1,r_cnt_F,r_cnt_S,kidx] = pred_deg
                # # get circular correlation coefficient and p values and store them
                # sig2[thispool,0], sig2[thispool,1] = circ_corr_pval(decoding2[:,thispool,0],decoding2[:,thispool,1],nComp,get_full=False)
                for s_cnt,s_stim_amp in enumerate(s_stim_amps):
                    this_stim_trial_idx = label_stim_strength_main == s_stim_amp
                    # get MAE
                    MAE2[s_cnt,r_cnt_F,r_cnt_S,thispool,kidx] = circ_MAE(tsty[this_stim_trial_idx]/180*pi, pred_deg[this_stim_trial_idx]/180*pi)*180/pi
    
    



        # for r_cnt_F in np.arange(len(r_stim_amps_F)):
    for r_cnt_S in np.arange(len(r_stim_amps_S)):
        for s_cnt,s_stim_amp in enumerate(s_stim_amps):
            if r_cnt_S == 0 & s_stim_amp == 10:
                # Stimulate + Unstimulated sub-network
                plt.figure(figsize=(4,4))
                plt.plot(MAE2[s_cnt,:,r_cnt_S,0,kidx],c='k',linewidth=2)
                plt.plot(np.mean(MAE2[s_cnt,:,r_cnt_S,1:,kidx],axis=1),'--k',linewidth=2)
                plt.legend(['Stimulated','Unstimulated'],frameon=False)
                plt.xlabel('Feature Gain Strength')
                plt.ylabel('MAE (deg)')
                plt.title(['K: ',str(rand_kappa),'S: ',str(r_stim_amps_S[r_cnt_S]),' SS: ', str(s_stim_amp)])
                plt.xticks(ticks=np.arange(len(r_stim_amps_F)),labels=r_stim_amps_F)
                plt.ylim([0,100])
                plt.show()
    
                
# # append decoding2 to the orig file
# newdic = {'decoding_sen2att':decoding2, 'MAE_sen2att':MAE2,'label_stim_strength_main':label_stim_strength_main}
# savemat('figure/result_FtoFxS_decoding.mat',newdic)

# labeldic = {'label_stim_strength_main':label_stim_strength_main}
# savemat('figure/result_FxS_label.mat',labeldic)
