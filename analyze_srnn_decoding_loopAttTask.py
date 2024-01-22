# SP
# Spiking network model to simulate top-down feature-based attention 

# Decode presented/attended stimulus from the spiking activities and plot decoding performance 

#%%
# change working directory to where all my functions are
# os.chdir('/mnt/neurocube/local/serenceslab/sunyoung/RNN/')

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.stats import circmean
import os.path
from os import path
# import SVC classifier
from sklearn.svm import SVC
# import metrics to compute accuracy
from sklearn.metrics import accuracy_score
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
from circ_reg_fromJohn import *
from circ_corr import *

nComp=1000 # for circ corr
# nRep = 10
reps = np.arange(1,17) # 1 - 16
nRep = len(reps)
# fileName = 'FBA_decoding'
fileName = 'FBA_decoding_shiftStep32'

# figure stuff
plotAllReps = 1
doSave = 1
figDPI = 300
fileFormat = 'eps'

dataFile = fileName+'_Rep1.npz'
data = np.load(dataFile)
N_stim_loc=data['N_stim_loc']
r_stim_amps=data['r_stim_amps']
r_stim_ratios=data['r_stim_ratios']
stim_strengths=data['stim_strengths']
N_trials=data['N_trials']
S_N_pools=data['S_N_pools']
S_N_neuron=data['S_N_neuron']

# #%% convert label (0-511) to degrees (0-2pi, 0-359)
# label_stim_loc_deg = (label_stim_loc/512*360).astype(int)
# label_stim_main_deg = (label_stim_main/512*360).astype(int)
# label_stim_main_bin = np.full(len(label_stim_main),np.nan)
# for lmi,lm in enumerate(np.unique(label_stim_main)):
#     label_stim_main_bin[label_stim_main==lm]=lmi
# label_stim_main_bin = label_stim_main_bin.astype(int)
# R_stim = 0 # one r_stim_ratio for now

#%% Analysis 1
# Train and test within localizer task for a sanity check
# define empty matrices to store predicted & actual values for each pool

# loop over network initialization reps and load saved file
decoding1 = np.full((N_trials*16,S_N_pools,2,nRep),np.nan) # test and predicted label for (all trials X pools)
MAE1 = np.full((S_N_pools,nRep),np.nan)
for rep_cnt, rep in enumerate(reps):
    dataFile = fileName+'_Rep'+str(rep)+'.npz'
    print('Loading '+dataFile)
    data = np.load(dataFile)
    
    # load variables
    S_fr_avg_loc=data['S_fr_avg_loc']
    label_stim_loc=data['label_stim_loc']
    label_pool_loc=data['label_pool_loc']
    label_trial_loc=data['label_trial_loc']
    S_fr_avg_main= data['S_fr_avg_main']
    label_stim_main= data['label_stim_main']
    label_trial_main=data['label_trial_main']
    N_stim_loc=data['N_stim_loc']
    r_stim_amps=data['r_stim_amps']
    r_stim_ratios=data['r_stim_ratios']
    stim_strengths=data['stim_strengths']
    N_trials=data['N_trials']
    S_N_pools=data['S_N_pools']
    S_N_neuron=data['S_N_neuron']
    # convert label (0-511) to degrees (0-2pi, 0-359)
    label_stim_loc_deg = (label_stim_loc/512*360).astype(int)
    
    # sig1 = np.full((S_N_pools,2),np.nan) # to bucket circ corr coef and p_vals
    
    for thispool in range(S_N_pools):
        pool_idx = np.arange(thispool*S_N_neuron,(thispool+1)*S_N_neuron) # indices of neurons in thispool
        poolX = S_fr_avg_loc[label_pool_loc==thispool,:] # subset trials when this pool was stimulated
        poolX = poolX[:,pool_idx] # subset neurons within this pool
        
        trial_idx_pool = label_trial_loc[label_pool_loc==thispool] # subset trials when this pool was stimulated
        y_pool = label_stim_loc_deg[label_pool_loc==thispool] # subset trials when this pool was stimulated
        for cv_iter in np.unique(trial_idx_pool): # loop over N_trials to leave out one trial per stimulus
            trnX = poolX[trial_idx_pool!=cv_iter,:]
            trny = y_pool[trial_idx_pool!=cv_iter]
            
            tstX = poolX[trial_idx_pool==cv_iter,:]
            tsty = y_pool[trial_idx_pool==cv_iter]
            
            my_model = compute_reg_weights(trnX,trny) # Circular regression with empirical bayes ridge
            pred = decode_reg(my_model,tstX)
            
            pred_deg = pred/np.pi*180 # convert radians back to degrees
            
            # store test & predicted labels and store them for visualization later
            pred_deg[np.abs(tsty-pred_deg)>180] = np.abs(pred_deg[np.abs(tsty-pred_deg)>180]-360)
            decoding1[trial_idx_pool==cv_iter,thispool,0,rep_cnt] = tsty
            decoding1[trial_idx_pool==cv_iter,thispool,1,rep_cnt] = pred_deg
            
        # get circular correlation coefficient and p values and store them
        # sig1[thispool,0], sig1[thispool,1] = circ_corr_pval(decoding1[:,thispool,0,rep],decoding1[:,thispool,1,rep],nComp,get_full=False)
        
        # get MAE
        MAE1[thispool,rep_cnt] = circ_MAE(decoding1[:,thispool,0,rep_cnt]/180*pi,decoding1[:,thispool,1,rep_cnt]/180*pi)*180/pi # TODO: check on this function

print(np.mean(np.mean(MAE1,axis=0)))

# plot predictions collapsed across all sub-networks
plt.figure(figsize=(4,4))
plt.scatter(decoding1[:,:,0,:].flatten(),decoding1[:,:,1,:].flatten(),c='k',alpha=0.01)
plt.xlabel('Presented Stimulus')
plt.ylabel('Predicted Stimulus')
plt.xticks(ticks=np.linspace(0,360,5))
plt.yticks(ticks=np.linspace(0,360,5))
plt.ylim([-10,370])
plt.xlim([-10,370])
if doSave==1:
    plt.savefig('fig1.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
plt.show()

#%% Analysis 2
# Train model with localizer task to classify presented stimulus (16-way)
# Test on main task data

MAE2 = np.full((len(r_stim_amps),S_N_pools,nRep),np.nan)
decoding2 = np.full((N_trials*2,S_N_pools,2,nRep),np.nan)
for r_cnt, R_stim in enumerate(r_stim_amps):
    for rep_cnt, rep in enumerate(reps):
        dataFile = fileName+'_Rep'+str(rep)+'.npz'
        print('Loading '+dataFile)
        data = np.load(dataFile)
        
        # load variables
        S_fr_avg_loc=data['S_fr_avg_loc']
        label_stim_loc=data['label_stim_loc']
        label_pool_loc=data['label_pool_loc']
        label_trial_loc=data['label_trial_loc']
        S_fr_avg_main= data['S_fr_avg_main']
        label_stim_main= data['label_stim_main']
        label_trial_main=data['label_trial_main']
        N_stim_loc=data['N_stim_loc']
        r_stim_amps=data['r_stim_amps']
        r_stim_ratios=data['r_stim_ratios']
        stim_strengths=data['stim_strengths']
        N_trials=data['N_trials']
        S_N_pools=data['S_N_pools']
        S_N_neuron=data['S_N_neuron']
        
        # convert label (0-511) to degrees (0-2pi, 0-359)
        label_stim_main_deg = (label_stim_main/512*360).astype(int)
        label_stim_main_bin = np.full(len(label_stim_main),np.nan)
        for lmi,lm in enumerate(np.unique(label_stim_main)):
            label_stim_main_bin[label_stim_main==lm]=lmi
        label_stim_main_bin = label_stim_main_bin.astype(int)
        R_ratio = 0 # one r_stim_ratio for now
    
    # sig2 = np.full((S_N_pools,2),np.nan) # to bucket circ corr coef and p_vals
        for thispool in np.arange(S_N_pools):
            pool_idx = np.arange(thispool*S_N_neuron,(thispool+1)*S_N_neuron) # indices of neurons in thispool
            
            poolX = S_fr_avg_loc[label_pool_loc==thispool,:] # subset trials when this pool was stimulated
            trnX = poolX[:,pool_idx] # subset neurons within this pool
            trny = label_stim_loc_deg[label_pool_loc==thispool] # subset trials when this pool was stimulated    
            
            tstX = S_fr_avg_main[:,pool_idx,r_cnt,R_ratio] # subset neurons within this pool
            tsty = label_stim_main_deg
            
            my_model = compute_reg_weights(trnX,trny) # Circular regression with empirical bayes ridge
            pred = decode_reg(my_model,tstX)
            
            pred_deg = pred/np.pi*180 # convert radians back to degrees
            
            # store test & predicted labels and store them for visualization later
            pred_deg[np.abs(tsty-pred_deg)>180] = np.abs(pred_deg[np.abs(tsty-pred_deg)>180]-360)
            decoding2[:,thispool,0,rep_cnt] = tsty
            decoding2[:,thispool,1,rep_cnt] = pred_deg
            # # get circular correlation coefficient and p values and store them
            # sig2[thispool,0], sig2[thispool,1] = circ_corr_pval(decoding2[:,thispool,0],decoding2[:,thispool,1],nComp,get_full=False)
            
            # get MAE
            MAE2[r_cnt,thispool,rep_cnt] = circ_MAE(tsty/180*pi,pred_deg/180*pi)*180/pi
            
    plt.figure(figsize=(4,4))
    plt.scatter(decoding2[:,0,0,:].flatten(),decoding2[:,0,1,:].flatten(),c='k',alpha=0.1)
    plt.xlabel('Attended Stimulus')
    plt.ylabel('Predicted Stimulus')
    plt.xticks(ticks=np.linspace(0,360,5))
    plt.yticks(ticks=np.linspace(0,360,5))
    plt.ylim([-10,370])
    plt.xlim([-10,370])
    if doSave==1:
        plt.savefig('fig2-scatter-stim-Rstim'+str(R_stim)+'.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
    plt.show()
    
    plt.figure(figsize=(4,4))
    plt.scatter(decoding2[:,1:,0,:].flatten(),decoding2[:,1:,1,:].flatten(),c='k',alpha=0.1)
    plt.xlabel('Attended Stimulus')
    plt.ylabel('Predicted Stimulus')
    plt.xticks(ticks=np.linspace(0,360,5))
    plt.yticks(ticks=np.linspace(0,360,5))
    plt.ylim([-10,370])
    plt.xlim([-10,370])
    if doSave==1:
        plt.savefig('fig2-scatter-unstim-Rstim'+str(R_stim)+'.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
    plt.show()

# Stimulated sub-network only
plt.figure(figsize=(4,4))
if plotAllReps==1:
    for rep_cnt, rep in enumerate(reps):
        plt.plot(MAE2[:,0,rep_cnt],linewidth=1,alpha=0.7)
plt.plot(np.mean(MAE2[:,0,:],axis=1),c='k',linewidth=2)
plt.xlabel('Gain Strength to Second Layer')
plt.ylabel('MAE (deg)')
plt.xticks(ticks=np.arange(len(r_stim_amps)),labels=r_stim_amps)
if doSave==1:
    if plotAllReps==0:
        plt.savefig('fig2-MAE-stim.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
    elif plotAllReps==1:
        plt.savefig('fig2-MAE-stim_allreps.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
plt.show()

# Stimulate + Unstimulated sub-network
plt.figure(figsize=(4,4))
if plotAllReps==1:
    plt.plot(np.mean(MAE2[:,0,:],axis=1),c='k',linewidth=2)
    plt.plot(np.mean(np.mean(MAE2[:,1:,:],axis=1),axis=1),'--k',linewidth=2)
    for rep_cnt, rep in enumerate(reps):
        plt.plot(MAE2[:,0,rep_cnt],linewidth=1,alpha=0.7)
        plt.plot(np.mean(MAE2[:,1:,rep_cnt],axis=1),linewidth=1,alpha=0.7,linestyle='--')
plt.plot(np.mean(MAE2[:,0,:],axis=1),c='k',linewidth=2)
plt.plot(np.mean(np.mean(MAE2[:,1:,:],axis=1),axis=1),'--k',linewidth=2)
plt.legend(['Stimulated','Unstimulated'],frameon=False)
plt.xlabel('Gain Strength to Second Layer')
plt.ylabel('MAE (deg)')
plt.xticks(ticks=np.arange(len(r_stim_amps)),labels=r_stim_amps)
# plt.ylim([,100])
if doSave==1:
    if plotAllReps==0:
        plt.savefig('fig2-MAE-stim+unstim.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
    elif plotAllReps==1:
        plt.savefig('fig2-MAE-stim+unstim_allreps.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
plt.show()

#%% Analysis 3 
# 2-way SVM

# Train model with main task data to classify attended stimulus (2-way)
# Test on main task data - cross validation (leave 2 trials out - one each for attended stim)
# -- this might need to be done with just usual svm?

dec_pool = np.full((S_N_pools,len(r_stim_amps),nRep),np.nan)
plt.figure(figsize=(4,4))
for rep_cnt, rep in enumerate(reps):
    dataFile = fileName+'_Rep'+str(rep)+'.npz'
    print('Loading '+dataFile)
    data = np.load(dataFile)
    
    # load variables
    S_fr_avg_loc=data['S_fr_avg_loc']
    label_stim_loc=data['label_stim_loc']
    label_pool_loc=data['label_pool_loc']
    label_trial_loc=data['label_trial_loc']
    S_fr_avg_main= data['S_fr_avg_main']
    label_stim_main= data['label_stim_main']
    label_trial_main=data['label_trial_main']
    N_stim_loc=data['N_stim_loc']
    r_stim_amps=data['r_stim_amps']
    r_stim_ratios=data['r_stim_ratios']
    stim_strengths=data['stim_strengths']
    N_trials=data['N_trials']
    S_N_pools=data['S_N_pools']
    S_N_neuron=data['S_N_neuron']
    
    # convert label (0-511) to degrees (0-2pi, 0-359)
    label_stim_main_deg = (label_stim_main/512*360).astype(int)
    label_stim_main_bin = np.full(len(label_stim_main),np.nan)
    for lmi,lm in enumerate(np.unique(label_stim_main)):
        label_stim_main_bin[label_stim_main==lm]=lmi
    label_stim_main_bin = label_stim_main_bin.astype(int)
    R_ratio = 0 # one r_stim_ratio for now
    
    dec_perf = np.full(N_trials,np.nan)
    for r_cnt, R_stim in enumerate(r_stim_amps):
        for thispool in range(S_N_pools):
            pool_idx = np.arange(thispool*S_N_neuron,(thispool+1)*S_N_neuron) # indices of neurons in thispool
            poolX = S_fr_avg_main[:,pool_idx,r_cnt,R_ratio] # subset neurons within this pool
            for cv_iter in np.unique(label_trial_main): # loop over N_trials to leave out one trial per stimulus
                # tst_idx = [stim_idx1[cv_iter],stim_idx2[cv_iter]]
                # trn_idx = np.delete(np.arange(len(label_main)),tst_idx)
                
                trnX = poolX[label_trial_main!=cv_iter,:] 
                trny = label_stim_main_bin[label_trial_main!=cv_iter] 
                
                tstX = poolX[label_trial_main==cv_iter,:] 
                tsty = label_stim_main_bin[label_trial_main==cv_iter]
                
                # instantiate classifier with default hyperparameters
                svc=SVC() 
                # fit classifier to training set
                svc.fit(trnX,trny)
                # make predictions on test set
                y_pred=svc.predict(tstX)
                # compute and print accuracy score
                dec_perf[int(cv_iter)] = accuracy_score(tsty, y_pred)
                # decoding1[label_trial_main==cv_iter,thispool,0] = accuracy_score(y_test, y_pred)
            dec_pool[thispool,r_cnt,rep_cnt] = np.mean(dec_perf)
# print(dec_pool) # decoding performance for predicting which stimulus (out of 2) was attended
    # plt.figure(figsize=(4,4))
#     plt.plot(dec_pool[0,:,rep_cnt]*100)
# plt.xlabel('Gain Strength to Second Layer')
# plt.ylabel('Decoding Performance (%)')
# plt.xticks(ticks=np.arange(len(r_stim_amps)),labels=r_stim_amps)
# plt.ylim([45, 105])
# plt.yticks(ticks=np.round(np.linspace(50,100,6)))
# # plt.legend(['Stimulated'],frameon=False)
# if doSave==1:
#     plt.savefig('fig3-dec-stim_allreps.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
# plt.show()
for plotAllReps in np.arange(2):
    # Stimulated sub-network only
    plt.figure(figsize=(4,4))
    if plotAllReps==1:
        plt.plot(np.mean(dec_pool[0,:,:],axis=1)*100,c='k',linewidth=1)
        for rep_cnt, rep in enumerate(reps):
            plt.plot(dec_pool[0,:,rep_cnt]*100,linewidth=1,alpha=0.7)
    plt.plot(np.mean(dec_pool[0,:,:],axis=1)*100,c='k',linewidth=2)
    plt.xlabel('Gain Strength to Second Layer')
    plt.ylabel('Decoding Performance (%)')
    plt.xticks(ticks=np.arange(len(r_stim_amps)),labels=r_stim_amps)
    plt.ylim([45, 105])
    plt.yticks(ticks=np.round(np.linspace(50,100,6)))
    plt.legend(['Stimulated'],frameon=False)
    if doSave==1:
        if plotAllReps==0:
            plt.savefig('fig3-dec-stim.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
        elif plotAllReps==1:
            plt.savefig('fig3-dec-stim_allreps.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
    plt.show()
    
    # Stimulated + Unstimulated together
    plt.figure(figsize=(4,4))
    if plotAllReps==1:
        plt.plot(np.mean(dec_pool[0,:,:],axis=1)*100,c='k',linewidth=1)
        plt.plot(np.mean(np.mean(dec_pool[1:,:,:],axis=2),axis=0)*100,'--k',linewidth=1)
        for rep_cnt, rep in enumerate(reps):
            plt.plot(dec_pool[0,:,rep_cnt]*100,linewidth=1,alpha=0.7)
            plt.plot(np.mean(dec_pool[1:,:,rep_cnt],axis=0)*100,linewidth=1,alpha=0.7,linestyle='--')
    plt.plot(np.mean(dec_pool[0,:,:],axis=1)*100,c='k',linewidth=2)
    plt.plot(np.mean(np.mean(dec_pool[1:,:,:],axis=2),axis=0)*100,'--k',linewidth=2)
    plt.xlabel('Gain Strength to Second Layer')
    plt.ylabel('Decoding Performance (%)')
    plt.xticks(ticks=np.arange(len(r_stim_amps)),labels=r_stim_amps)
    plt.ylim([45, 105])
    plt.yticks(ticks=np.round(np.linspace(50,100,6)))
    plt.legend(['Stimulated','Unstimulated'],frameon=False)
    if doSave==1:
        if plotAllReps==0:
            plt.savefig('fig3-dec-stim+unstim.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
        elif plotAllReps==1:
            plt.savefig('fig3-dec-stim+unstim_allreps.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
    
        # plt.savefig('fig3-dec-unstim.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
        # plt.savefig('fig3-dec-unstim_allreps.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
    plt.show()

#%% Analysis 4
# Train on one unstimulated pool and test on another unstimulated pool
# all combinations
# plt.figure(figsize=(4,4))
dec_x_pool = np.full((S_N_pools-1,len(r_stim_amps),nRep),np.nan)
for trn_pool_cnt,trn_pool in enumerate(np.arange(1,S_N_pools)):
    tst_pool = np.delete(np.arange(1,S_N_pools),np.arange(1,S_N_pools)==trn_pool)
    for rep_cnt, rep in enumerate(reps):
        dataFile = fileName+'_Rep'+str(rep)+'.npz'
        print('Loading '+dataFile)
        data = np.load(dataFile)
        
        # load variables
        S_fr_avg_loc=data['S_fr_avg_loc']
        label_stim_loc=data['label_stim_loc']
        label_pool_loc=data['label_pool_loc']
        label_trial_loc=data['label_trial_loc']
        S_fr_avg_main= data['S_fr_avg_main']
        label_stim_main= data['label_stim_main']
        label_trial_main=data['label_trial_main']
        N_stim_loc=data['N_stim_loc']
        r_stim_amps=data['r_stim_amps']
        r_stim_ratios=data['r_stim_ratios']
        stim_strengths=data['stim_strengths']
        N_trials=data['N_trials']
        S_N_pools=data['S_N_pools']
        S_N_neuron=data['S_N_neuron']
        
        # convert label (0-511) to degrees (0-2pi, 0-359)
        label_stim_main_deg = (label_stim_main/512*360).astype(int)
        label_stim_main_bin = np.full(len(label_stim_main),np.nan)
        for lmi,lm in enumerate(np.unique(label_stim_main)):
            label_stim_main_bin[label_stim_main==lm]=lmi
        label_stim_main_bin = label_stim_main_bin.astype(int)
        R_ratio = 0 # one r_stim_ratio for now
        
        pool_idx_trn = np.arange(trn_pool*S_N_neuron,(trn_pool+1)*S_N_neuron) # indices of neurons in thispool
        for r_cnt, R_stim in enumerate(r_stim_amps):
            trnX = S_fr_avg_main[:,pool_idx_trn,r_cnt,0]
            trny = label_stim_main_bin
            tsty = label_stim_main_bin
            
            # instantiate classifier with default hyperparameters
            svc=SVC() 
            # fit classifier to training set
            svc.fit(trnX,trny)
            
            dec_temp = np.full(len(tst_pool),np.nan)
            for pi, thispool in enumerate(tst_pool):
                pool_idx_tst = np.arange(thispool*S_N_neuron,(thispool+1)*S_N_neuron) # indices of neurons in thispool
                tstX = S_fr_avg_main[:,pool_idx_tst,r_cnt,0] # subset neurons within this pool
                
                # make predictions on test set
                y_pred=svc.predict(tstX)
                # compute and print accuracy score
                dec_temp[pi] = accuracy_score(tsty, y_pred) # get decoding accuracy for each test pool
            dec_x_pool[trn_pool_cnt,r_cnt,rep_cnt] = np.mean(dec_temp) # average and store for each training pool
            
        # print(dec_x_pool)
        
    # plt.figure(figsize=(4,4))
#         plt.plot(dec_x_pool[trn_pool_cnt,:,rep_cnt]*100)
# plt.xlabel('Gain Strength to Second Layer')
# plt.ylabel('Decoding Performance (%)')
# plt.xticks(ticks=np.arange(len(r_stim_amps)),labels=r_stim_amps)
# # plt.legend(reps)
# plt.ylim([45, 105])
# plt.yticks(ticks=np.round(np.linspace(50,100,6)))
# plt.show()
for plotAllReps in np.arange(2):
    plt.figure(figsize=(4,4))
    if plotAllReps==1:
        for rep_cnt, rep in enumerate(reps):
            plt.plot(np.mean(dec_x_pool[:,:,rep_cnt],axis=0)*100,linewidth=1,alpha=0.7)
    plt.plot(np.mean(np.mean(dec_x_pool,axis=2),axis=0)*100,c='k',linewidth=2)
    plt.xlabel('Gain Strength to Second Layer')
    plt.ylabel('Decoding Performance (%)')
    plt.xticks(ticks=np.arange(len(r_stim_amps)),labels=r_stim_amps)
    plt.ylim([45, 105])
    plt.yticks(ticks=np.round(np.linspace(50,100,6)))
    if doSave==1:
        if plotAllReps==0:
            plt.savefig('fig4-dec.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
        elif plotAllReps==1:
            plt.savefig('fig4-dec_allreps.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
    plt.show()


for rep_cnt, rep in enumerate(reps):
    for trn_pool_cnt,trn_pool in enumerate(np.arange(1,S_N_pools)):
        plt.plot(dec_x_pool[trn_pool_cnt,:,rep_cnt]*100)
    plt.xlabel('Gain Strength to Second Layer')
    plt.ylabel('Decoding Performance (%)')
    plt.xticks(ticks=np.arange(len(r_stim_amps)),labels=r_stim_amps)
    plt.legend(reps)
    plt.ylim([45, 105])
    plt.yticks(ticks=np.round(np.linspace(50,100,6)))
    plt.show()
