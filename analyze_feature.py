#%%
# change working directory to where all my functions are
# os.chdir('/mnt/neurocube/local/serenceslab/sunyoung/RNN/results')

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


os.chdir('/mnt/neurocube/local/serenceslab/sunyoung/RNN/')

from circ_reg_fromJohn import *
from circ_corr import *

nComp=1000 # for circ corr
# nRep = 10
save_all_kpa=0
doPlot=0

# reps = [1]
#%%
# figure stuff
plotAllReps = 0
doSave = 0
figDPI = 300
fileFormat = 'eps'

# kappas = [0,0.1,0.2,0.3,0.4]
kappas = [0,0.1]

for rand_kappa in kappas:
# rand_kappa = 0
    nRep = 10
    reps = np.arange(1,nRep+1) # 1 - 16
    
    # os.chdir('/mnt/neurocube/local/serenceslab/sunyoung/RNN/')
    # load one file to grab basic parameters
    dataFile = 'results/F_kappa-'+str(rand_kappa)+'_seed-1.npz'
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
    
    N_trials_sensory=data['N_trials_sensory']
    N_trials_attention=data['N_trials_attention']
    
    S_N_pools=data['S_N_pools']
    S_N_neuron=data['S_N_neuron']
    
    # combine data for higher attention gain sims
    dataFile = 'results/F_higher-Rstim_kappa-'+str(rand_kappa)+'_seed-1.npz'    
    data = np.load(dataFile)    
    r_stim_amps = np.append(r_stim_amps,data['r_stim_amps'])
    S_fr_avg_main = np.concatenate((S_fr_avg_main,data['S_fr_avg_main']),axis=2)
    
    
    # #%% Analysis 1
    # # Train and test within localizer task for a sanity check
    # # define empty matrices to store predicted & actual values for each pool
    
    # # loop over network initialization reps and load saved file
    # decoding1 = np.full((N_trials_sensory*N_stim_loc,S_N_pools,2,nRep),np.nan) # test and predicted label for (all trials X pools)
    # MAE1 = np.full((S_N_pools,nRep),np.nan)
    # for rep_cnt, rep in enumerate(reps):
    #     dataFile = 'results/F_kappa-'+str(rand_kappa)+'_seed-'+str(rep)+'.npz'
    #     print('Loading '+dataFile)
    #     data = np.load(dataFile)
        
    #     # load variables
    #     S_fr_avg_loc=data['S_fr_avg_loc']
    #     label_stim_loc=data['label_stim_loc']
    #     label_pool_loc=data['label_pool_loc']
    #     label_trial_loc=data['label_trial_loc']
        
    #     N_stim_loc=data['N_stim_loc']
        
    #     stim_strengths=data['stim_strengths']
        
    #     N_trials_sensory=data['N_trials_sensory']
        
    #     S_N_pools=data['S_N_pools']
    #     S_N_neuron=data['S_N_neuron']
        
    #     # convert label (0-511) to degrees (0-2pi, 0-359)
    #     label_stim_loc_deg = (label_stim_loc/S_N_neuron*360).astype(int)
        
    #     # sig1 = np.full((S_N_pools,2),np.nan) # to bucket circ corr coef and p_vals
        
    #     for thispool in range(S_N_pools):
    #         pool_idx = np.arange(thispool*S_N_neuron,(thispool+1)*S_N_neuron) # indices of neurons in thispool
    #         poolX = S_fr_avg_loc[label_pool_loc==thispool,:] # subset trials when this pool was stimulated
    #         poolX = poolX[:,pool_idx] # subset neurons within this pool
            
    #         trial_idx_pool = label_trial_loc[label_pool_loc==thispool] # subset trials when this pool was stimulated
    #         y_pool = label_stim_loc_deg[label_pool_loc==thispool] # subset trials when this pool was stimulated
    #         for cv_iter in np.unique(trial_idx_pool): # loop over N_trials to leave out one trial per stimulus
    #             trnX = poolX[trial_idx_pool!=cv_iter,:]
    #             trny = y_pool[trial_idx_pool!=cv_iter]
                
    #             tstX = poolX[trial_idx_pool==cv_iter,:]
    #             tsty = y_pool[trial_idx_pool==cv_iter]
                
    #             my_model = compute_reg_weights(trnX,trny) # Circular regression with empirical bayes ridge
    #             pred = decode_reg(my_model,tstX)
                
    #             pred_deg = pred/np.pi*180 # convert radians back to degrees
                
    #             # store test & predicted labels and store them for visualization later
    #             pred_deg[np.abs(tsty-pred_deg)>180] = np.abs(pred_deg[np.abs(tsty-pred_deg)>180]-360)
    #             decoding1[trial_idx_pool==cv_iter,thispool,0,rep_cnt] = tsty
    #             decoding1[trial_idx_pool==cv_iter,thispool,1,rep_cnt] = pred_deg
                
    #         # get circular correlation coefficient and p values and store them
    #         # sig1[thispool,0], sig1[thispool,1] = circ_corr_pval(decoding1[:,thispool,0,rep],decoding1[:,thispool,1,rep],nComp,get_full=False)
            
    #         # get MAE
    #         MAE1[thispool,rep_cnt] = circ_MAE(decoding1[:,thispool,0,rep_cnt]/180*pi,decoding1[:,thispool,1,rep_cnt]/180*pi)*180/pi # TODO: check on this function
    
    # print(np.mean(np.mean(MAE1,axis=0)))
    
    # # plot predictions collapsed across all sub-networks
    # if doPlot:
    #     plt.figure(figsize=(4,4))
    #     plt.scatter(decoding1[:,:,0,:].flatten(),decoding1[:,:,1,:].flatten(),c='k',alpha=0.01)
    #     plt.xlabel('Presented Stimulus')
    #     plt.ylabel('Predicted Stimulus')
    #     plt.xticks(ticks=np.linspace(0,360,5))
    #     plt.yticks(ticks=np.linspace(0,360,5))
    #     plt.ylim([-10,370])
    #     plt.xlim([-10,370])
    #     if doSave==1:
    #         plt.savefig('fig1.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
    #     plt.show()
    
    #%% Analysis 2
    # Train model with localizer task to classify presented stimulus (16-way)
    # Test on main task data
    
    MAE2 = np.full((len(r_stim_amps),S_N_pools,nRep,len(stim_strengths)),np.nan)
    MAE2_deg = np.full((len(r_stim_amps),S_N_pools,nRep,len(stim_strengths)),np.nan)
    decoding2 = np.full((len(label_trial_main),S_N_pools,2,nRep,len(r_stim_amps)),np.nan)
    
    
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
            
        for r_cnt, R_stim in enumerate(r_stim_amps):
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
                # pred_deg[np.abs(tsty-pred_deg)>180] = np.abs(pred_deg[np.abs(tsty-pred_deg)>180]-360)
                decoding2[:,thispool,0,rep_cnt,r_cnt] = tsty
                decoding2[:,thispool,1,rep_cnt,r_cnt] = pred_deg
                # s_stim_amp = 10
                # this_stim_trial_idx = label_stim_strength_main == s_stim_amp
                # d[np.abs(d)>pi]-= 2*pi*np.sign(d[np.abs(d)>pi])
                # MAE = np.mean(abs(wrapRad(y_hat-y)))
                
                # temp = pred_deg[this_stim_trial_idx] - tsty[this_stim_trial_idx]
                # temp[np.abs(temp)>180] -= 2*180*np.sign(temp[np.abs(temp)>180])
                # print(np.mean(np.abs(temp)))
                
                # plt.hist(pred_deg)
                # plt.title(thispool)
                # plt.show()
                # # get circular correlation coefficient and p values and store them
                # sig2[thispool,0], sig2[thispool,1] = circ_corr_pval(decoding2[:,thispool,0],decoding2[:,thispool,1],nComp,get_full=False)
                
                
                # get MAE - NEW: calculate MAE separately for different stim strength levels
                for s_cnt, s_stim_amp in enumerate(stim_strengths):
                # for ss_cnt in np.arange(len(stim_strengths)):
                    this_stim_trial_idx = label_stim_strength_main == s_stim_amp
                    # d[np.abs(d)>pi]-= 2*pi*np.sign(d[np.abs(d)>pi])
                    # MAE = np.mean(abs(wrapRad(y_hat-y)))
                    
                    temp = pred_deg[this_stim_trial_idx] - tsty[this_stim_trial_idx]
                    temp[np.abs(temp)>180] -= 2*180*np.sign(temp[np.abs(temp)>180])
                    MAE2_deg[r_cnt,thispool,rep_cnt,s_cnt] = np.mean(np.abs(temp))
                    MAE2[r_cnt,thispool,rep_cnt,s_cnt] = circ_MAE(tsty[this_stim_trial_idx]/180*pi, pred_deg[this_stim_trial_idx]/180*pi)*180/pi
                    if np.round(np.mean(np.abs(temp)),2) != np.round(MAE2[r_cnt,thispool,rep_cnt,s_cnt],2):
                        print(MAE2_deg[r_cnt,thispool,rep_cnt,s_cnt],MAE2[r_cnt,thispool,rep_cnt,s_cnt])
                        raise ValueError('MAE Values do not match')
    # datdic = {'MAE_sen2att':MAE2,'MAE_deg_sen2att':MAE2_deg}
    # # os.chdir('/mnt/neurocube/local/serenceslab/sunyoung/RNN/figure')
    
    # savemat('figure/test_result_F_kappa-'+str(rand_kappa)+'.mat',datdic)
    # print('saved figure/test_result_F_kappa-'+str(rand_kappa)+'.mat')
    

        if doPlot:
            plt.figure(figsize=(4,4))
            plt.scatter(decoding2[:,0,0,:,r_cnt].flatten(),decoding2[:,0,1,:,r_cnt].flatten(),c='k',alpha=0.1)
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
            plt.scatter(decoding2[:,1:,0,:,r_cnt].flatten(),decoding2[:,1:,1,:,r_cnt].flatten(),c='k',alpha=0.1)
            plt.xlabel('Attended Stimulus')
            plt.ylabel('Predicted Stimulus')
            plt.xticks(ticks=np.linspace(0,360,5))
            plt.yticks(ticks=np.linspace(0,360,5))
            plt.ylim([-10,370])
            plt.xlim([-10,370])
            if doSave==1:
                plt.savefig('fig2-scatter-unstim-Rstim'+str(R_stim)+'.'+fileFormat, dpi=figDPI,bbox_inches='tight',format=fileFormat)
            plt.show()
        
    # append decoding2 to the orig file
    # newdic = {'decoding_sen2att':decoding2}
    # savemat('result_kappa-'+str(rand_kappa)+'.mat',newdic)
    
    if doPlot:
        # Stimulated sub-network only
        plt.figure(figsize=(4,4))
        if plotAllReps==1:
            for rep_cnt, rep in enumerate(reps):
                plt.plot(MAE2[:,0,rep_cnt,10],linewidth=1,alpha=0.7)
        plt.plot(np.mean(MAE2[:,0,:,8],axis=1),c='k',linewidth=2)
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
        
    # datdic = {'MAE_sen2att':MAE2,'MAE_deg_sen2att':MAE2_deg}
    # # os.chdir('/mnt/neurocube/local/serenceslab/sunyoung/RNN/figure')
    
    # savemat('figure/test_result_F_kappa-'+str(rand_kappa)+'.mat',datdic)
    # print('saved figure/test_result_F_kappa-'+str(rand_kappa)+'.mat')
    #%% Analysis 3 
    # 2-way SVM: NEW - train/test separately for each stimulus strength levels
    
    # Train model with main task data to classify attended stimulus (2-way)
    # Test on main task data - cross validation (leave 2 trials out - one each for attended stim)
    # -- this might need to be done with just usual svm?
    
    # plt.figure(figsize=(4,4))
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
        
        print(N_trials_sensory,N_trials_attention)
        
        if rep_cnt == 0:
            dec_pool = np.full((S_N_pools,len(r_stim_amps),nRep,len(stim_strengths)),np.nan)
        
        # convert label (0-511) to degrees (0-2pi, 0-359)
        label_stim_main_deg = (label_stim_main/S_N_neuron*360).astype(int)
        label_stim_main_bin = np.full(len(label_stim_main),np.nan)
        for lmi,lm in enumerate(np.unique(label_stim_main)):
            label_stim_main_bin[label_stim_main==lm]=lmi
        label_stim_main_bin = label_stim_main_bin.astype(int)
        R_ratio = 0 # one r_stim_ratio for now
        
        for s_cnt, s_stim_amp in enumerate(stim_strengths):
            dec_perf = np.full(N_trials_attention,np.nan)
            for r_cnt, R_stim in enumerate(r_stim_amps):
                for thispool in range(S_N_pools):
                    pool_idx = np.arange(thispool*S_N_neuron,(thispool+1)*S_N_neuron) # indices of neurons in thispool
                    poolX = S_fr_avg_main[:,pool_idx,r_cnt,R_ratio] # subset neurons within this pool
                    for cv_iter in np.unique(label_trial_main): # loop over N_trials to leave out one trial per stimulus
                        # tst_idx = [stim_idx1[cv_iter],stim_idx2[cv_iter]]
                        # trn_idx = np.delete(np.arange(len(label_main)),tst_idx)
                        trn_idx = (label_trial_main!=cv_iter)*(label_stim_strength_main==s_stim_amp)
                        tst_idx = (label_trial_main==cv_iter)*(label_stim_strength_main==s_stim_amp)
                        
                        trnX = poolX[trn_idx,:] 
                        trny = label_stim_main_bin[trn_idx] 
                        
                        tstX = poolX[tst_idx,:] 
                        tsty = label_stim_main_bin[tst_idx]
                        
                        # instantiate classifier with default hyperparameters
                        svc=SVC() 
                        # fit classifier to training set
                        svc.fit(trnX,trny)
                        # make predictions on test set
                        y_pred=svc.predict(tstX)
                        # compute and print accuracy score
                        dec_perf[int(cv_iter)] = accuracy_score(tsty, y_pred)
                        # decoding1[label_trial_main==cv_iter,thispool,0] = accuracy_score(y_test, y_pred)
                    dec_pool[thispool,r_cnt,rep_cnt,s_cnt] = np.mean(dec_perf)
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
    if doPlot:
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
    
    #%% Analysis 4 NEW - train/test separately for each stimulus strength levels
    # Train on one unstimulated pool and test on another unstimulated pool
    # all combinations
    # plt.figure(figsize=(4,4))
    dec_x_pool = np.full((S_N_pools-1,len(r_stim_amps),nRep,len(stim_strengths)),np.nan)
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
            
            
            # convert label (0-511) to degrees (0-2pi, 0-359)
            label_stim_main_deg = (label_stim_main/S_N_neuron*360).astype(int)
            label_stim_main_bin = np.full(len(label_stim_main),np.nan)
            for lmi,lm in enumerate(np.unique(label_stim_main)):
                label_stim_main_bin[label_stim_main==lm]=lmi
            label_stim_main_bin = label_stim_main_bin.astype(int)
            R_ratio = 0 # one r_stim_ratio for now
            
            for s_cnt, s_stim_amp in enumerate(stim_strengths):
                ss_trial_idx = label_stim_strength_main==s_stim_amp
                pool_idx_trn = np.arange(trn_pool*S_N_neuron,(trn_pool+1)*S_N_neuron) # indices of neurons in thispool
                for r_cnt, R_stim in enumerate(r_stim_amps):
                    ss_temp=S_fr_avg_main[ss_trial_idx,:,:,:]
                    trnX = ss_temp[:,pool_idx_trn,r_cnt,0]
                    trny = label_stim_main_bin[ss_trial_idx]
                    tsty = label_stim_main_bin[ss_trial_idx]
                    
                    # instantiate classifier with default hyperparameters
                    svc=SVC() 
                    # fit classifier to training set
                    svc.fit(trnX,trny)
                    
                    dec_temp = np.full(len(tst_pool),np.nan)
                    for pi, thispool in enumerate(tst_pool):
                        pool_idx_tst = np.arange(thispool*S_N_neuron,(thispool+1)*S_N_neuron) # indices of neurons in thispool
                        tstX = ss_temp[:,pool_idx_tst,r_cnt,0] # subset neurons within this pool
                        
                        # make predictions on test set
                        y_pred=svc.predict(tstX)
                        # compute and print accuracy score
                        dec_temp[pi] = accuracy_score(tsty, y_pred) # get decoding accuracy for each test pool
                    dec_x_pool[trn_pool_cnt,r_cnt,rep_cnt,s_cnt] = np.mean(dec_temp) # average and store for each training pool
                
    if doPlot:
        for plotAllReps in np.arange(2):
            plt.figure(figsize=(4,4))
            if plotAllReps==1:
                for rep_cnt, rep in enumerate(reps):
                    plt.plot(np.mean(dec_x_pool[:,:,rep_cnt],axis=0)*100,linewidth=1,alpha=0.7)
            plt.plot(np.mean(np.mean(dec_x_pool,axis=2),axis=0)*100,c='k',linewidth=2)
            plt.xlabel('Gain Strength to Second Layer')
            plt.ylabel('Decoding Performance (%)')
            plt.xticks(ticks=np.arange(len(r_stim_amps)),labels=r_stim_amps)
            # plt.ylim([-5, 105])
            plt.ylim([45, 105])
            # plt.yticks(ticks=np.round(np.linspace(50,100,6)))
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
            plt.ylim([-5, 105])
            # plt.yticks(ticks=np.round(np.linspace(50,100,6)))
            plt.show()
    
    #%% save out summarized data for plotting
   
    
    datdic = {'decoding_sen2att':decoding2, 'MAE_sen2att':MAE2,'decoding_att2att':dec_pool,'decoding_att2att_xpool':dec_x_pool,
              'reps':reps, 'rand_kappa':rand_kappa, 'dataFile':dataFile, 
              'N_stim_loc':N_stim_loc, 'r_stim_amps':r_stim_amps, 'r_stim_ratios':r_stim_ratios,'stim_strengths':stim_strengths,
              'N_trials_sensory':N_trials_sensory, 'N_trials_attention':N_trials_attention,
              'label_stim_strength_main':label_stim_strength_main,'MAE_deg_sen2att':MAE2_deg}
    # os.chdir('/mnt/neurocube/local/serenceslab/sunyoung/RNN/figure')

    savemat('figure/result_F_kappa-'+str(rand_kappa)+'_new_deg.mat',datdic)
    print('saved figure/result_F_kappa-'+str(rand_kappa)+'_new_deg.mat')
    
    # labeldic = {'label_stim_strength_main':label_stim_strength_main}
    # savemat('figure/result_F_label.mat',labeldic)
    



    
    #%% Plot CRF for different kappas
    
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
# for different kappa levels, it makes more sense to compare stimulated sub-network and unstimulated sub-network
# as opposed to comparing attended vs unattended feature response within the stimulated sub-network
os.chdir('/mnt/neurocube/local/serenceslab/sunyoung/RNN/')
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


#%% Plotting attended vs unattended CRF for different kappas
os.chdir('/mnt/neurocube/local/serenceslab/sunyoung/RNN/')
N_reps = 10
reps = np.arange(1,N_reps+1)
kappas = [0,0.1,0.2,0.3,0.4]
s_stim_amps = np.arange(20) # strength of bottom-up stimulus
avg_FR_stim_att_kappa = np.full((len(s_stim_amps),len(r_stim_amps),N_reps,5),np.nan)
avg_FR_stim_unatt_kappa = np.full((len(s_stim_amps),len(r_stim_amps),N_reps,5),np.nan)
cont_gain_att_kappa = np.full((len(r_stim_amps),N_reps,5),np.nan)
cont_gain_unatt_kappa = np.full((len(r_stim_amps),N_reps,5),np.nan)
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
        R_ratio = 0 # one r_stim_ratio for now
        
        # averaging FRs of neurons that prefer the stimulus
        N_stim_neuron = 2 #  94 by function
        
        if rep_cnt == 0:
            avg_FR_stim_att = np.full((len(s_stim_amps),len(r_stim_amps),N_reps),np.nan)
            cont_gain_att = np.full((len(r_stim_amps),N_reps),np.nan)
            
            avg_FR_stim_unatt = np.full((len(s_stim_amps),len(r_stim_amps),N_reps),np.nan)
            cont_gain_unatt = np.full((len(r_stim_amps),N_reps),np.nan)
        
        colors = plt.cm.viridis(np.linspace(1,0,len(r_stim_amps)))
        # cmap = plt.get_cmap('viridis')
    
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
