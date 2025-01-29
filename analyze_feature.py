#%% analyze_feature.py
# Takes npz output files and runs decoding analyses for feature-based 
# attention simulations. Saves out mat files for plotting figures in Matlab.

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

from scipy.special import i0, i1 

from scipy.io import savemat

# set working directory to where the helper scripts exist
os.chdir('/mnt/neurocube/local/serenceslab/sunyoung/RNN/')
from helper_codes import *

nComp=1000 # for circ corr

# figure stuff
doPlot=0
plotAllReps = 0 # plot data for different network initializations
saveFig = 0 # save figures- optional since we're gonna plot again in Matlab
figDPI = 300
fileFormat = 'eps'


#%%

nRep = 10 # network initializations
reps = np.arange(nRep)+1 # 1 - 10

kappas = [0,0.1,0.2,0.3,0.4] # K for connectivity randomness

# looping over all K values
for rand_kappa in kappas:
    #%% Analysis 1
    # Train and test the circular regression model within sensory task for a sanity check
    # This takes a while to run...
    
    # loop over network initialization reps and load saved file
    for rep_cnt, rep in enumerate(reps):
        dataFile = 'results/F_kappa-'+str(rand_kappa)+'_seed-'+str(rep)+'.npz'
        print('Loading '+dataFile)
        data = np.load(dataFile)
        
        # load variables
        S_fr_avg_loc=data['S_fr_avg_loc']
        label_stim_loc=data['label_stim_loc']
        label_pool_loc=data['label_pool_loc']
        label_trial_loc=data['label_trial_loc']
        
        N_stim_loc=data['N_stim_loc']
        
        stim_strengths=data['stim_strengths']
        
        N_trials_sensory=data['N_trials_sensory']
        
        S_N_pools=data['S_N_pools']
        S_N_neuron=data['S_N_neuron']
        
        # convert label (0-511) to degrees (0-2pi, 0-359)
        label_stim_loc_deg = (label_stim_loc/S_N_neuron*360).astype(int)
        
        # define empty matrices to store predicted & actual values for each sub-network
        if rep_cnt == 0:
            decoding1 = np.full((N_trials_sensory*N_stim_loc,S_N_pools,2,nRep),np.nan) # test and predicted label for (all trials X pools)
            MAE1 = np.full((S_N_pools,nRep),np.nan)
        
        # loop over each sub-network
        for thispool in range(S_N_pools):
            pool_idx = np.arange(thispool*S_N_neuron,(thispool+1)*S_N_neuron) # indices of neurons in thispool
            poolX = S_fr_avg_loc[label_pool_loc==thispool,:] # subset trials when this pool was stimulated
            poolX = poolX[:,pool_idx] # subset neurons within this pool
            
            trial_idx_pool = label_trial_loc[label_pool_loc==thispool] # trial index
            y_pool = label_stim_loc_deg[label_pool_loc==thispool] # actual stimulus label
            
            # loop over trials to leave out one trial per stimulus
            for cv_iter in np.unique(trial_idx_pool): 
                trnX = poolX[trial_idx_pool!=cv_iter,:] # train data
                trny = y_pool[trial_idx_pool!=cv_iter] # train label
                
                tstX = poolX[trial_idx_pool==cv_iter,:] # test data
                tsty = y_pool[trial_idx_pool==cv_iter] # test label
                
                # Circular regression with empirical bayes ridge (from helper_codes.py)
                my_model = compute_reg_weights(trnX,trny) 
                pred = decode_reg(my_model,tstX)
                
                pred_deg = pred/np.pi*180 # convert radians back to degrees
                
                # store actual & predicted labels for visualization
                pred_deg[np.abs(tsty-pred_deg)>180] = np.abs(
                    pred_deg[np.abs(tsty-pred_deg)>180]-360) # wrapping degrees around the circular space
                decoding1[trial_idx_pool==cv_iter,thispool,0,rep_cnt] = tsty
                decoding1[trial_idx_pool==cv_iter,thispool,1,rep_cnt] = pred_deg
                
            # calculate mean absolute error (MAE) (from helper_codes.py)
            MAE1[thispool,rep_cnt] = circ_MAE(
                decoding1[:,thispool,0,rep_cnt]/180*pi,
                decoding1[:,thispool,1,rep_cnt]/180*pi)*180/pi
    
    # print average MAE for this kappa
    print(np.mean(np.mean(MAE1,axis=0)))
    
    # plot actual & predicted labels collapsed across all sub-networks
    if doPlot:
        plt.figure(figsize=(4,4))
        plt.scatter(decoding1[:,:,0,:].flatten(),decoding1[:,:,1,:].flatten(),c='k',alpha=0.01)
        plt.xlabel('Presented Stimulus')
        plt.ylabel('Predicted Stimulus')
        plt.xticks(ticks=np.linspace(0,360,5))
        plt.yticks(ticks=np.linspace(0,360,5))
        plt.ylim([-10,370])
        plt.xlim([-10,370])
        if saveFig==1:
            plt.savefig('fig1.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
        plt.show()
    
    #%% Analysis 2
    # Train the circular regression model with sensory task and test on attention task
    
    # loop over network initialization reps and load saved file
    for rep_cnt, rep in enumerate(reps):
        dataFile = 'results/F_kappa-'+str(rand_kappa)+'_seed-'+str(rep)+'.npz'
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
        label_stim_strength_main=data['label_stim_strength_main']
        
        N_stim_loc=data['N_stim_loc']
        
        r_stim_amps=data['r_stim_amps']
        r_stim_ratios=data['r_stim_ratios']
        stim_strengths=data['stim_strengths']
        
        N_trials_sensory=data['N_trials_sensory']
        N_trials_attention=data['N_trials_attention']
        
        S_N_pools=data['S_N_pools']
        S_N_neuron=data['S_N_neuron']
        
        # combine data for higher attention gain sims
        dataFile = 'results/F_higher-Rstim_kappa-'+str(rand_kappa)+'_seed-'+str(rep)+'.npz'    
        data = np.load(dataFile)    
        r_stim_amps = np.append(r_stim_amps,data['r_stim_amps'])
        print(r_stim_amps)
        S_fr_avg_main = np.concatenate((S_fr_avg_main,data['S_fr_avg_main']),axis=2)
        
        # convert label (0-511) to degrees (0-2pi, 0-359)
        label_stim_main_deg = (label_stim_main/S_N_neuron*360).astype(int)
        label_stim_main_bin = np.full(len(label_stim_main),np.nan)
        for lmi,lm in enumerate(np.unique(label_stim_main)):
            label_stim_main_bin[label_stim_main==lm]=lmi
        label_stim_main_bin = label_stim_main_bin.astype(int)
        R_ratio = 0 # one r_stim_ratio for now
        
        label_stim_loc_deg = (label_stim_loc/S_N_neuron*360).astype(int)
        
        # define empty matrices to store predicted & actual values for each sub-network
        if rep_cnt == 0:
            decoding2 = np.full((len(label_trial_main),S_N_pools,2,nRep,len(r_stim_amps)),np.nan)
            MAE2 = np.full((len(r_stim_amps),S_N_pools,nRep,len(stim_strengths)),np.nan)
            
        # loop over top-down gain strengths
        for r_cnt, R_stim in enumerate(r_stim_amps):
            
            # loop over each sub-network
            for thispool in np.arange(S_N_pools):
                pool_idx = np.arange(thispool*S_N_neuron,(thispool+1)*S_N_neuron) # indices of neurons in thispool
                
                poolX = S_fr_avg_loc[label_pool_loc==thispool,:] # subset trials when this sub-network was stimulated
                trnX = poolX[:,pool_idx] # subset data for this sub-network - sensory task
                trny = label_stim_loc_deg[label_pool_loc==thispool] # train label - sensory task
                
                tstX = S_fr_avg_main[:,pool_idx,r_cnt,R_ratio] # test data - attention task
                tsty = label_stim_main_deg # test label - attention task
                
                # Circular regression with empirical bayes ridge (from helper_codes.py)
                my_model = compute_reg_weights(trnX,trny)
                pred = decode_reg(my_model,tstX)
                
                pred_deg = pred/np.pi*180 # convert radians back to degrees
                
                # store test & predicted labels
                pred_deg[np.abs(tsty-pred_deg)>180] = np.abs(
                    pred_deg[np.abs(tsty-pred_deg)>180]-360) # wrapping degrees around the circular space
                decoding2[:,thispool,0,rep_cnt,r_cnt] = tsty
                decoding2[:,thispool,1,rep_cnt,r_cnt] = pred_deg
                
                # calculate MAE separately for different stim strength levels
                for s_cnt, s_stim_amp in enumerate(stim_strengths):
                    this_stim_trial_idx = label_stim_strength_main == s_stim_amp
                    MAE2[r_cnt,thispool,rep_cnt,s_cnt] = circ_MAE(
                        tsty[this_stim_trial_idx]/180*pi, pred_deg[this_stim_trial_idx]/180*pi)*180/pi
        
        # plot actual and predicted labels
        if doPlot:
            plt.figure(figsize=(4,4))
            plt.scatter(decoding2[:,0,0,:,r_cnt].flatten(),decoding2[:,0,1,:,r_cnt].flatten(),c='k',alpha=0.1)
            plt.xlabel('Attended Stimulus')
            plt.ylabel('Predicted Stimulus')
            plt.xticks(ticks=np.linspace(0,360,5))
            plt.yticks(ticks=np.linspace(0,360,5))
            plt.ylim([-10,370])
            plt.xlim([-10,370])
            if saveFig==1:
                plt.savefig('fig2-scatter-stim-Rstim'+str(R_stim)+'.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
            plt.show()
            
            plt.figure(figsize=(4,4))
            plt.scatter(decoding2[:,1:,0,:,r_cnt].flatten(),decoding2[:,1:,1,:,r_cnt].flatten(),c='k',alpha=0.1)
            plt.xlabel('Attended Stimulus')
            plt.ylabel('Predicted Stimulus')
            plt.xticks(ticks=np.linspace(0,360,5))
            plt.yticks(ticks=np.linspace(0,360,5))
            plt.ylim([-10,370])
            plt.xlim([-10,370])
            if saveFig==1:
                plt.savefig('fig2-scatter-unstim-Rstim'+str(R_stim)+'.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
            plt.show()
    
    # plot average MAE across all stim strengths
    if doPlot:
        avgMAE = np.mean(MAE2,3)
        
        # Stimulated sub-network only
        plt.figure(figsize=(4,4))
        if plotAllReps==1:
            for rep_cnt, rep in enumerate(reps):
                plt.plot(avgMAE[:,0,rep_cnt],linewidth=1,alpha=0.7)
        plt.plot(np.mean(avgMAE[:,0,:],axis=1),c='k',linewidth=2)
        plt.xlabel('Gain Strength to Second Layer')
        plt.ylabel('MAE (deg)')
        plt.xticks(ticks=np.arange(len(r_stim_amps)),labels=r_stim_amps)
        if saveFig==1:
            if plotAllReps==0:
                plt.savefig('fig2-MAE-stim.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
            elif plotAllReps==1:
                plt.savefig('fig2-MAE-stim_allreps.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
        plt.show()
        
        # Stimulate + Unstimulated sub-network
        plt.figure(figsize=(4,4))
        if plotAllReps==1:
            for rep_cnt, rep in enumerate(reps):
                plt.plot(avgMAE[:,0,rep_cnt],linewidth=1,alpha=0.7)
                plt.plot(np.mean(avgMAE[:,1:,rep_cnt],axis=1),linewidth=1,alpha=0.7,linestyle='--')
        plt.plot(np.mean(avgMAE[:,0,:],axis=1),c='k',linewidth=2)
        plt.plot(np.mean(np.mean(avgMAE[:,1:,:],axis=1),axis=1),'--k',linewidth=2)
        plt.legend(['Stimulated','Unstimulated'],frameon=False)
        plt.xlabel('Gain Strength to Second Layer')
        plt.ylabel('MAE (deg)')
        plt.xticks(ticks=np.arange(len(r_stim_amps)),labels=r_stim_amps)
        if saveFig==1:
            if plotAllReps==0:
                plt.savefig('fig2-MAE-stim+unstim.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
            elif plotAllReps==1:
                plt.savefig('fig2-MAE-stim+unstim_allreps.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
        plt.show()
        
    #%% Analysis 3 
    # Train/test the SVM on attention task (2-way classification on which of the two stimuli was attended)
    # train/test separately for each stimulus strength levels
    # cross validation: leave 2 trials out - one each for attended stim
    
    for rep_cnt, rep in enumerate(reps):
        dataFile = 'results/F_kappa-'+str(rand_kappa)+'_seed-'+str(rep)+'.npz'
        print('Loading '+dataFile)
        data = np.load(dataFile)
        
        # load variables    
        S_fr_avg_main= data['S_fr_avg_main']
        label_stim_main= data['label_stim_main']
        label_trial_main=data['label_trial_main']
        label_stim_strength_main=data['label_stim_strength_main']
        
        r_stim_amps=data['r_stim_amps']
        r_stim_ratios=data['r_stim_ratios']
        stim_strengths=data['stim_strengths']
        
        N_trials_attention=data['N_trials_attention']
        
        S_N_pools=data['S_N_pools']
        S_N_neuron=data['S_N_neuron']
        
        # combine data for higher attention gain sims
        dataFile = 'results/F_higher-Rstim_kappa-'+str(rand_kappa)+'_seed-'+str(rep)+'.npz'    
        
        data = np.load(dataFile)    
        r_stim_amps = np.append(r_stim_amps,data['r_stim_amps'])
        S_fr_avg_main = np.concatenate((S_fr_avg_main,data['S_fr_avg_main']),axis=2)
        
        # convert label (0-511) to binary labels (0,1)
        label_stim_main_bin = np.full(len(label_stim_main),np.nan)
        for lmi,lm in enumerate(np.unique(label_stim_main)):
            label_stim_main_bin[label_stim_main==lm]=lmi
        label_stim_main_bin = label_stim_main_bin.astype(int)
        R_ratio = 0 # one r_stim_ratio for now
        
        # define empty matrices to store predicted & actual values for each sub-network
        if rep_cnt == 0:
            dec_pool = np.full((S_N_pools,len(r_stim_amps),nRep,len(stim_strengths)),np.nan)
        
        # loop over stimulus strengths
        for s_cnt, s_stim_amp in enumerate(stim_strengths):
            # empty matrix to store accuracy
            dec_perf = np.full(N_trials_attention,np.nan)
            # loop over top-down gain strengths
            for r_cnt, R_stim in enumerate(r_stim_amps):
                # loop over sub-networks
                for thispool in range(S_N_pools):
                    pool_idx = np.arange(thispool*S_N_neuron,(thispool+1)*S_N_neuron) # indices of neurons in thispool
                    poolX = S_fr_avg_main[:,pool_idx,r_cnt,R_ratio] # subset neurons within this pool
                    # loop over trial index to leave out one trial per stimulus
                    for cv_iter in np.unique(label_trial_main): 
                        # define trial indices for train/test
                        trn_idx = (label_trial_main!=cv_iter)*(label_stim_strength_main==s_stim_amp)
                        tst_idx = (label_trial_main==cv_iter)*(label_stim_strength_main==s_stim_amp)
                        
                        trnX = poolX[trn_idx,:]  # train data
                        trny = label_stim_main_bin[trn_idx] # train label
                        
                        tstX = poolX[tst_idx,:] # test data
                        tsty = label_stim_main_bin[tst_idx] # test label
                        
                        # instantiate classifier with default hyperparameters
                        svc=SVC() 
                        # fit classifier to training set
                        svc.fit(trnX,trny)
                        # make predictions on test set
                        y_pred=svc.predict(tstX)
                        # compute and store accuracy score
                        dec_perf[int(cv_iter)] = accuracy_score(tsty, y_pred)
                    # store average decoding accuracy across cross-validation loop
                    dec_pool[thispool,r_cnt,rep_cnt,s_cnt] = np.mean(dec_perf)
    
    # plot average decoding accuracy across stimulus strengths
    if doPlot:
        avg_dec = np.mean(dec_pool,3)
        # Stimulated sub-network only
        plt.figure(figsize=(4,4))
        if plotAllReps==1:
            for rep_cnt, rep in enumerate(reps):
                plt.plot(avg_dec[0,:,rep_cnt]*100,linewidth=1,alpha=0.7)
        plt.plot(np.mean(avg_dec[0,:,:],axis=1)*100,c='k',linewidth=2)
        plt.xlabel('Gain Strength to Second Layer')
        plt.ylabel('Decoding Performance (%)')
        plt.xticks(ticks=np.arange(len(r_stim_amps)),labels=r_stim_amps)
        plt.ylim([45, 105])
        plt.yticks(ticks=np.round(np.linspace(50,100,6)))
        plt.legend(['Stimulated'],frameon=False)
        if saveFig==1:
            if plotAllReps==0:
                plt.savefig('fig3-dec-stim.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
            elif plotAllReps==1:
                plt.savefig('fig3-dec-stim_allreps.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
        plt.show()
        
        # Stimulated + Unstimulated together
        plt.figure(figsize=(4,4))
        if plotAllReps==1:
            for rep_cnt, rep in enumerate(reps):
                plt.plot(avg_dec[0,:,rep_cnt]*100,linewidth=1,alpha=0.7)
                plt.plot(np.mean(avg_dec[1:,:,rep_cnt],axis=0)*100,linewidth=1,alpha=0.7,linestyle='--')
        plt.plot(np.mean(avg_dec[0,:,:],axis=1)*100,c='k',linewidth=2)
        plt.plot(np.mean(np.mean(avg_dec[1:,:,:],axis=2),axis=0)*100,'--k',linewidth=2)
        plt.xlabel('Gain Strength to Second Layer')
        plt.ylabel('Decoding Performance (%)')
        plt.xticks(ticks=np.arange(len(r_stim_amps)),labels=r_stim_amps)
        plt.ylim([45, 105])
        plt.yticks(ticks=np.round(np.linspace(50,100,6)))
        plt.legend(['Stimulated','Unstimulated'],frameon=False)
        if saveFig==1:
            if plotAllReps==0:
                plt.savefig('fig3-dec-stim+unstim.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
            elif plotAllReps==1:
                plt.savefig('fig3-dec-stim+unstim_allreps.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
        plt.show()
    
    #%% Analysis 4
    # Train on one unstimulated sub-network and test on another unstimulated sub-network
    # Train/test separately for each stimulus strength levels
    # Loop over all pairs of train/test sub-networks
    
    # loop over unstimulated sub-networks as training set
    for trn_pool_cnt,trn_pool in enumerate(np.arange(1,S_N_pools)):
        tst_pool = np.delete(np.arange(1,S_N_pools),np.arange(1,S_N_pools)==trn_pool)
        for rep_cnt, rep in enumerate(reps):
            dataFile = 'results/F_kappa-'+str(rand_kappa)+'_seed-'+str(rep)+'.npz'
            print('Loading '+dataFile)
            data = np.load(dataFile)
            
            # load variables        
            S_fr_avg_main= data['S_fr_avg_main']
            label_stim_main= data['label_stim_main']
            label_trial_main=data['label_trial_main']
            label_stim_strength_main=data['label_stim_strength_main']
            
            N_stim_loc=data['N_stim_loc']
            
            r_stim_amps=data['r_stim_amps']
            r_stim_ratios=data['r_stim_ratios']
            stim_strengths=data['stim_strengths']
            
            N_trials_attention=data['N_trials_attention']
            
            S_N_pools=data['S_N_pools']
            S_N_neuron=data['S_N_neuron']
            
            # combine data for higher attention gain sims
            dataFile = 'results/F_higher-Rstim_kappa-'+str(rand_kappa)+'_seed-'+str(rep)+'.npz'    
            data = np.load(dataFile)    
            r_stim_amps = np.append(r_stim_amps,data['r_stim_amps'])
            S_fr_avg_main = np.concatenate((S_fr_avg_main,data['S_fr_avg_main']),axis=2)
            
            # convert label (0-511) to binary labels (0,1)
            label_stim_main_bin = np.full(len(label_stim_main),np.nan)
            for lmi,lm in enumerate(np.unique(label_stim_main)):
                label_stim_main_bin[label_stim_main==lm]=lmi
            label_stim_main_bin = label_stim_main_bin.astype(int)
            R_ratio = 0 # one r_stim_ratio for now
            
            # define empty matrices to store predicted & actual values for each sub-network
            if rep_cnt == 0:
                dec_x_pool = np.full((S_N_pools-1,len(r_stim_amps),nRep,len(stim_strengths)),np.nan)
            
            # loop over stimulus strengths
            for s_cnt, s_stim_amp in enumerate(stim_strengths):
                # define trial indices for this stim strength
                ss_trial_idx = label_stim_strength_main==s_stim_amp
                pool_idx_trn = np.arange(trn_pool*S_N_neuron,(trn_pool+1)*S_N_neuron) # indices of neurons in trn_pool
                # loop over top-down gain strengths
                for r_cnt, R_stim in enumerate(r_stim_amps):
                    data_thisSS=S_fr_avg_main[ss_trial_idx,:,:,:] # subset data for this stim strength
                    trnX = data_thisSS[:,pool_idx_trn,r_cnt,0] # train data
                    trny = label_stim_main_bin[ss_trial_idx] # train label
                    tsty = label_stim_main_bin[ss_trial_idx] # test label
                    
                    # instantiate classifier with default hyperparameters
                    svc=SVC() 
                    # fit classifier to training set
                    svc.fit(trnX,trny)
                    
                    # empty matrix for decoding accuracies across all test sub-networks
                    dec_temp = np.full(len(tst_pool),np.nan)
                    
                    # loop over unstimulated sub-networks as test set
                    for p_idx, thispool in enumerate(tst_pool):
                        pool_idx_tst = np.arange(thispool*S_N_neuron,(thispool+1)*S_N_neuron) # indices of neurons in thispool
                        tstX = data_thisSS[:,pool_idx_tst,r_cnt,0] # subset neurons within this pool
                        
                        # make predictions on test set
                        y_pred=svc.predict(tstX)
                        # compute and store accuracy score
                        dec_temp[p_idx] = accuracy_score(tsty, y_pred) # get decoding accuracy for each test sub-network
                    # average decoding accuracies across test sub-networks
                    dec_x_pool[trn_pool_cnt,r_cnt,rep_cnt,s_cnt] = np.mean(dec_temp) 
    # plot average decoding accuracy across stimulus strengths        
    if doPlot:
        avg_dec = np.mean(dec_x_pool,3)
        plt.figure(figsize=(4,4))
        if plotAllReps==1:
            for rep_cnt, rep in enumerate(reps):
                plt.plot(np.mean(avg_dec[:,:,rep_cnt],axis=0)*100,linewidth=1,alpha=0.7)
        plt.plot(np.mean(np.mean(avg_dec,axis=2),axis=0)*100,c='k',linewidth=2)
        plt.xlabel('Gain Strength to Second Layer')
        plt.ylabel('Decoding Performance (%)')
        plt.xticks(ticks=np.arange(len(r_stim_amps)),labels=r_stim_amps)
        plt.ylim([-5, 105])
        plt.ylim([45, 105])
        plt.yticks(ticks=np.round(np.linspace(50,100,6)))
        if saveFig==1:
            if plotAllReps==0:
                plt.savefig('fig4-dec.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
            elif plotAllReps==1:
                plt.savefig('fig4-dec_allreps.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
        plt.show()
    
    #%% save out summarized data for plotting
    datdic = {'decoding_sen2att':decoding2, 'MAE_sen2att':MAE2,'decoding_att2att':dec_pool,'decoding_att2att_xpool':dec_x_pool,
              'reps':reps, 'rand_kappa':rand_kappa, 'dataFile':dataFile, 
              'N_stim_loc':N_stim_loc, 'r_stim_amps':r_stim_amps, 'r_stim_ratios':r_stim_ratios,'stim_strengths':stim_strengths,
              'N_trials_sensory':N_trials_sensory, 'N_trials_attention':N_trials_attention,
              'label_stim_strength_main':label_stim_strength_main}

    savemat('figure/result_F_kappa-'+str(rand_kappa)+'.mat',datdic)
    print('saved figure/result_F_kappa-'+str(rand_kappa)+'.mat')
    
#%% Plot CRFs
    
# When comparing across different kappa levels, it makes more sense to compare 
# stimulated vs unstimulated sub-network instead of comparing 
# attended vs unattended feature response within the stimulated sub-network.
# We'll calculate and save both - stim vs unstim first, then att vs unatt.

if save_all_kappa == 1:
    N_reps = 10
    reps = np.arange(1,N_reps+1)
    avg_FR_stim_kappa = np.full((len(stim_strengths),len(r_stim_amps),S_N_pools,N_reps,5),np.nan)
    cont_gain_kappa = np.full((len(r_stim_amps),N_reps,5),np.nan)
    kappas = [0,0.1,0.2,0.3,0.4]
    for k_cnt, rand_kappa in enumerate(kappas):
        for rep_cnt, rep in enumerate(reps):
            dataFile = 'results/F_kappa-'+str(rand_kappa)+'_seed-'+str(rep)+'.npz'
            data = np.load(dataFile)
            
            # load variables
            S_fr_avg_main= data['S_fr_avg_main']
            label_stim_main= data['label_stim_main']
            label_trial_main=data['label_trial_main']
            label_stim_strength_main=data['label_stim_strength_main']
            
            N_stim_loc=data['N_stim_loc']
            
            r_stim_amps=data['r_stim_amps']
            r_stim_ratios=data['r_stim_ratios']
            s_stim_amps=data['stim_strengths']
            
            N_trials_attention=data['N_trials_attention']
            
            S_N_pools=data['S_N_pools']
            S_N_neuron=data['S_N_neuron']
            
            # combine data for higher attention gain sims
            dataFile = 'results/F_higher-Rstim_kappa-'+str(rand_kappa)+'_seed-'+str(rep)+'.npz'    
            data = np.load(dataFile)    
            r_stim_amps = np.append(r_stim_amps,data['r_stim_amps'])
            S_fr_avg_main = np.concatenate((S_fr_avg_main,data['S_fr_avg_main']),axis=2)
            
            #% All plots in one
            avg_FR = np.full((S_N_neuron,2,len(s_stim_amps)),np.nan)
            R_ratio = 0 # one r_stim_ratio for now
            
            # averaging FRs of neurons that prefer the stimulus
            N_stim_neuron = 2 #  94 by function
            Stim_idx = (np.arange(N_stim_neuron)+(S_N_neuron/2-N_stim_neuron/2)).astype(int)
            # avg_FR_stim = np.full((len(s_stim_amps),2),np.nan)
            avg_FR_stim_trial = np.full((N_trials_attention,len(s_stim_amps),2),np.nan)
            
            if rep_cnt == 0:
                avg_FR_stim = np.full((len(s_stim_amps),len(r_stim_amps),S_N_pools,N_reps),np.nan)
                cont_gain = np.full((len(r_stim_amps),N_reps),np.nan)
                
                # avg_FR_stim_unatt = np.full((len(s_stim_amps),len(r_stim_amps),N_reps),np.nan)
                # cont_gain_unatt = np.full((len(r_stim_amps),N_reps),np.nan)
            
            colors = plt.cm.viridis(np.linspace(1,0,len(r_stim_amps)))
            
            for r_cnt, R_stim in enumerate(r_stim_amps):
                for ss_cnt, S_stim in enumerate(s_stim_amps):    
                    # trial_idx = (label_stim_strength_main==S_stim) # average separately for trials for attention to stim vs unstim pool
                    trial_idx_label = np.argwhere(label_stim_strength_main==S_stim) # location indices to loop through
                    # doing this the long way... so that it works with whatever stim label
                    for pool in np.arange(S_N_pools):
                        temp = np.full((len(trial_idx_label)),np.nan) # store average response for each trial (because stim center changes based on the trial)
                        for t_cnt, tt in enumerate(trial_idx_label):
                            Stim_idx = (np.arange(N_stim_neuron)+(label_stim_main[tt]-N_stim_neuron/2)+pool*S_N_neuron).astype(int)
                            # if t_cnt==0:
                                # print(Stim_idx)
                            temp[t_cnt] = np.mean(S_fr_avg_main[tt,Stim_idx,r_cnt,0],axis=0)
                        avg_FR_stim[ss_cnt,r_cnt,pool,rep_cnt] = np.mean(temp)
                # curve fitting
                params, rss = fitCRF2(s_stim_amps/20,avg_FR_stim[:,r_cnt,0,rep_cnt],0)
                
                if np.isnan(rss):
                    cont_gain[r_cnt,rep_cnt] = nan
                else:
                    cont_gain[r_cnt,rep_cnt] = params[1]
                
                    # for pool in np.arange(S_N_pools):
                    #     avg_FR_stim[ss_cnt,r_cnt,pool,rep_cnt] = \
                    #         np.mean(np.mean(S_fr_avg_main[trial_idx,:,r_cnt,0][:,Stim_idx+pool*S_N_neuron],axis=0),axis=0) # trial-average of FR
        avg_FR_stim_kappa[:,:,:,:,k_cnt] = avg_FR_stim # save out
        cont_gain_kappa[:,:,k_cnt] = cont_gain
        
    # plot
    for k_cnt, rand_kappa in enumerate(kappas):
        avg_FR_stim = avg_FR_stim_kappa[:,:,:,:,k_cnt]
        avg_FR_stim_att = avg_FR_stim[:,:,0,:]
        avg_FR_stim_unatt = np.mean(avg_FR_stim[:,:,1:,:],axis=2)
        repavg_FR_stim_att = np.mean(avg_FR_stim_att,axis=2)
        se_FR_stim_att = np.std(avg_FR_stim_att,axis=2)/np.sqrt(N_reps)
        repavg_FR_stim_unatt = np.mean(avg_FR_stim_unatt,axis=2)
        se_FR_stim_unatt = np.std(avg_FR_stim_unatt,axis=2)/np.sqrt(N_reps)
        
        # line1 = np.full((len(r_stim_amps_F)),np.nan)
        
        # for r_cnt_S, R_stim_S in enumerate(r_stim_amps_S): # loop over spatial attention first
        plt.figure(figsize=(8,5))
        for r_cnt, R_stim in enumerate(r_stim_amps[0:9]):
            plt.plot(repavg_FR_stim_att[:,r_cnt],color=colors[r_cnt,:],marker='o',linestyle='solid',fillstyle='none',label=str(R_stim))
            yerr = se_FR_stim_att[:,r_cnt]
            plt.fill_between(s_stim_amps, repavg_FR_stim_att[:,r_cnt] - yerr, repavg_FR_stim_att[:,r_cnt] + yerr, color=colors[r_cnt,:],alpha=0.3)
            
            plt.plot(repavg_FR_stim_unatt[:,r_cnt],color=colors[r_cnt,:],marker='o',linestyle='dotted',fillstyle='none')
            yerr = se_FR_stim_unatt[:,r_cnt]
            plt.fill_between(s_stim_amps, repavg_FR_stim_unatt[:,r_cnt] - yerr, repavg_FR_stim_unatt[:,r_cnt] + yerr, color=colors[r_cnt,:],alpha=0.3)
           
            # plt.fill_between(x, y - yerr, y + yerr, alpha=0.3)
            # for rep_cnt, rep in enumerate(reps):
            #     plt.plot(avg_FR_stim[:,r_cnt_F,r_cnt_S,rep_cnt],color=colors[r_cnt_F,:])
         
        plt.legend()
        plt.xticks(s_stim_amps,s_stim_amps)
        # plt.title('Spatial Att. Gain : '+str(R_stim_S))
        plt.xlabel('Stimulus Strength')
        plt.ylabel('Avg FR for preferred neurons')
        plt.title('kappa = '+str(rand_kappa))
        plt.show()
        
        # plot a ratio between stim and unstim
        
        ratio_attunatt = np.log(avg_FR_stim_att/avg_FR_stim_unatt)
        repavg_FR_ratio = np.mean(ratio_attunatt,axis=2)
        se_FR_ratio = np.std(ratio_attunatt,axis=2)/np.sqrt(N_reps)
        plt.figure(figsize=(8,5))
        for r_cnt, R_stim in enumerate(r_stim_amps[0:9]):
            plt.plot(repavg_FR_ratio[:,r_cnt],color=colors[r_cnt,:],marker='o',linestyle='solid',fillstyle='none',label=str(R_stim))
            yerr = se_FR_ratio[:,r_cnt]
            plt.fill_between(s_stim_amps, repavg_FR_ratio[:,r_cnt] - yerr, repavg_FR_ratio[:,r_cnt] + yerr, color=colors[r_cnt,:],alpha=0.3)
        
        plt.xticks(s_stim_amps,s_stim_amps)
        plt.xlabel('Stimulus Strength')
        plt.ylabel('log(Stim/Unstim)')
        plt.title('kappa = '+str(rand_kappa))
        plt.show()
            
            
    
    # append decoding2 to the orig file
    # savemat('result_kappa-'+str(rand_kappa)+'.mat',newdic,appendmat=True)
    
    # save kappa file separately
    # os.chdir('/mnt/neurocube/local/serenceslab/sunyoung/RNN/figure')
    newdic = {'avg_FR_stim_kappa':avg_FR_stim_kappa,'cont_gain_kappa':cont_gain_kappa}
    savemat('figure/result_F_unstimPool.mat',newdic)
    print('Saved figure/result_F_unstimPool.mat')


#%% Plotting attended vs unattended CRF for kappa levels
os.chdir('/mnt/neurocube/local/serenceslab/sunyoung/RNN/')
N_reps = 10
reps = np.arange(N_reps)+1
kappas = [0,0.1,0.2,0.3,0.4]
s_stim_amps = np.arange(20) # strength of bottom-up stimulus

avg_FR_stim_att_kappa = np.full((len(s_stim_amps),len(r_stim_amps),N_reps,5),np.nan)
avg_FR_stim_unatt_kappa = np.full((len(s_stim_amps),len(r_stim_amps),N_reps,5),np.nan)
cont_gain_att_kappa = np.full((len(r_stim_amps),N_reps,5),np.nan)
cont_gain_unatt_kappa = np.full((len(r_stim_amps),N_reps,5),np.nan)

# loop over kappas
for k_cnt, rand_kappa in enumerate(kappas):
    
    # loop over network initializations
    for rep_cnt, rep in enumerate(reps):
        dataFile = 'results/F_kappa-'+str(rand_kappa)+'_seed-'+str(rep)+'.npz'
        data = np.load(dataFile)
        
        # load variables
        S_fr_avg_main= data['S_fr_avg_main']
        label_stim_main= data['label_stim_main']
        label_trial_main=data['label_trial_main']
        label_stim_strength_main=data['label_stim_strength_main']
        
        N_stim_loc=data['N_stim_loc']
        
        r_stim_amps=data['r_stim_amps']
        r_stim_ratios=data['r_stim_ratios']
        s_stim_amps=data['stim_strengths']
        
        N_trials_attention=data['N_trials_attention']
        
        S_N_pools=data['S_N_pools']
        S_N_neuron=data['S_N_neuron']
        
        # combine data for higher attention gain sims
        dataFile = 'results/F_higher-Rstim_kappa-'+str(rand_kappa)+'_seed-'+str(rep)+'.npz'    
        data = np.load(dataFile)    
        r_stim_amps = np.append(r_stim_amps,data['r_stim_amps'])
        S_fr_avg_main = np.concatenate((S_fr_avg_main,data['S_fr_avg_main']),axis=2)
        
        #% All plots in one
        R_ratio = 0 # one r_stim_ratio for now
        
        # averaging FRs of neurons that prefer the stimulus
        N_stim_neuron = 2 #  94 by function
        
        # define empty matrices
        if rep_cnt == 0:
            avg_FR_stim_att = np.full((len(s_stim_amps),len(r_stim_amps),N_reps),np.nan)
            cont_gain_att = np.full((len(r_stim_amps),N_reps),np.nan)
            
            avg_FR_stim_unatt = np.full((len(s_stim_amps),len(r_stim_amps),N_reps),np.nan)
            cont_gain_unatt = np.full((len(r_stim_amps),N_reps),np.nan)
        
        colors = plt.cm.viridis(np.linspace(1,0,len(r_stim_amps)))
    
        for r_cnt, R_stim in enumerate(r_stim_amps):
            for ss_cnt, S_stim in enumerate(s_stim_amps):    
                att_pool = 0
                # for att_pool in np.arange(1): # only for attended sub-network only for now
                    # trial_idx = label_stim_strength_main==S_stim
                trial_idx = np.argwhere((label_stim_strength_main==S_stim)) # average separately for trials for attention to stim vs unstim pool
                
                # attended stimulus
                temp = np.full((len(trial_idx)),np.nan)
                for tidx,trial in enumerate(trial_idx):
                    feature_idx = label_stim_main[trial]
                    Stim_idx = (np.arange(N_stim_neuron)+(feature_idx-N_stim_neuron/2)).astype(int)
                    
                    temp[tidx] = np.mean(S_fr_avg_main[trial,Stim_idx,r_cnt],axis=0)
                avg_FR_stim_att[ss_cnt,r_cnt,rep_cnt] = np.mean(temp,axis=0) # trial-average of FR
                
                # unattended stimulus
                temp = np.full((len(trial_idx)),np.nan)
                for tidx,trial in enumerate(trial_idx):
                    feature_idx = np.mod(label_stim_main[trial]+S_N_neuron/2,S_N_neuron)
                    Stim_idx = (np.arange(N_stim_neuron)+(feature_idx-N_stim_neuron/2)).astype(int)
                    
                    temp[tidx] = np.mean(S_fr_avg_main[trial,Stim_idx,r_cnt],axis=0)
                avg_FR_stim_unatt[ss_cnt,r_cnt,rep_cnt] = np.mean(temp,axis=0) # trial-average of FR
                
            # init_params, rss = gridfitCRF(s_stim_amps/20,avg_FR_stim[:,r_cnt_F,r_cnt_S,rep_cnt],1)
            params, rss = fitCRF2(s_stim_amps/20,avg_FR_stim_att[:,r_cnt,rep_cnt],0)
            # params, rss = fitCRF(s_stim_amps/20,avg_FR_stim[:,r_cnt_F,r_cnt_S,rep_cnt],init_params,1)
            if np.isnan(rss):
                cont_gain_att[r_cnt,rep_cnt] = nan
            else:
                cont_gain_att[r_cnt,rep_cnt] = params[1]
                
            params, rss = fitCRF2(s_stim_amps/20,avg_FR_stim_unatt[:,r_cnt,rep_cnt],0)
            # params, rss = fitCRF(s_stim_amps/20,avg_FR_stim[:,r_cnt_F,r_cnt_S,rep_cnt],init_params,1)
            if np.isnan(rss):
                cont_gain_unatt[r_cnt,rep_cnt] = nan
            else:
                cont_gain_unatt[r_cnt,rep_cnt] = params[1]
                
    avg_FR_stim_att_kappa[:,:,:,k_cnt] = avg_FR_stim_att # save out
    avg_FR_stim_unatt_kappa[:,:,:,k_cnt] = avg_FR_stim_unatt
    cont_gain_att_kappa[:,:,k_cnt] = cont_gain_att
    cont_gain_unatt_kappa[:,:,k_cnt] = cont_gain_unatt
    
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
        
        plt.plot(repavg_FR_stim_unatt[:,r_cnt],color=colors[r_cnt,:],marker='o',linestyle='dotted',fillstyle='none')
        yerr = se_FR_stim_unatt[:,r_cnt]
        plt.fill_between(s_stim_amps, repavg_FR_stim_unatt[:,r_cnt] - yerr, repavg_FR_stim_unatt[:,r_cnt] + yerr, color=colors[r_cnt,:],alpha=0.3)
       
        # plt.fill_between(x, y - yerr, y + yerr, alpha=0.3)
        # for rep_cnt, rep in enumerate(reps):
        #     plt.plot(avg_FR_stim[:,r_cnt_F,r_cnt_S,rep_cnt],color=colors[r_cnt_F,:])
 
    plt.legend()
    plt.xticks(s_stim_amps,s_stim_amps)
    plt.title('kappa : '+str(rand_kappa))
    plt.xlabel('Stimulus Strength')
    plt.ylabel('Avg FR for preferred neurons')
    plt.show()
        
    #% Contrast gain
    plt.figure(figsize=(8,5))
    repavg_cont_gain = np.nanmean(cont_gain_att,axis=1)
    se_cont_gain = np.std(cont_gain_att,axis=1)/np.sqrt(N_reps)
    # for r_cnt_S, R_stim_S in enumerate(r_stim_amps_S):
    plt.plot(r_stim_amps,repavg_cont_gain)
    yerr = se_cont_gain
    plt.fill_between(r_stim_amps, repavg_cont_gain - yerr, repavg_cont_gain + yerr,alpha=0.3)
        
    repavg_cont_gain = np.nanmean(cont_gain_unatt,axis=1)
    se_cont_gain = np.std(cont_gain_unatt,axis=1)/np.sqrt(N_reps)
    # for r_cnt_S, R_stim_S in enumerate(r_stim_amps_S):
    plt.plot(r_stim_amps,repavg_cont_gain,linestyle='dotted')
    # plt.plot(r_stim_amps,repavg_cont_gain,color='grey',linestyle='dotted')
    plt.fill_between(r_stim_amps, repavg_cont_gain - yerr, repavg_cont_gain + yerr,alpha=0.3)
        
    
        # for rep_cnt, rep in enumerate(reps):
        #     plt.plot(cont_gain[:,r_cnt_S,rep_cnt],color=colors[r_cnt_S,:])
    
    plt.xlabel('Feature Attention Strength')
    plt.ylabel('Contrast Gain')
    plt.legend()
    plt.show()
    
newdic = {'avg_FR_stim_att_kappa':avg_FR_stim_att_kappa,'avg_FR_stim_unatt_kappa':avg_FR_stim_unatt_kappa,
          'cont_gain_att_kappa':cont_gain_att_kappa,'cont_gain_unatt_kappa':cont_gain_unatt_kappa}
savemat('figure/result_F_unattStim.mat',newdic)
print('Saved figure/result_F_unattStim.mat')
