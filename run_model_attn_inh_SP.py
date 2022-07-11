from brian2 import *
from rand_attn_inh_SP import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.stats import circmean
import os.path
from os import path
#%%
#-------------------------
# Set rnd seed so that can repeat model...important! deterministic from here on out...
#-------------------------
rndSeed = 0

#-------------------------
# number of neurons in sensory and random layers
# S_ means sensory, R_ means random, (S|R)_N means number of sensory or random neurons
#-------------------------
S_N_neuron = 512
R_N_neuron = 1024

#-------------------------
# number of pools of sensory neurons
#-------------------------
S_N_pools = 8

#-------------------------
# tau: https://www.frontiersin.org/articles/10.3389/fncir.2020.615626/full
#-------------------------
S_tau = 0.01 * second   # for sensory layer
R_tau = 0.01 * second   # for random layer

#-------------------------
# bias
#-------------------------
S_bias = 0              # for sensory layer
R_bias = 0              # for random layer

#-------------------------
# slope of non-linear transfer function 
#-------------------------
S_tf_slope = 0.25       # for sensory layer- important for dynamic range....4 puts it into attractor land
R_tf_slope = 0.25        # for random layer

#-------------------------
# seed initial synapse range
#-------------------------
S_InitSynRange = .01    # for sensory layer
R_InitSynRange = .01    # for random layer

#-------------------------
# time step to save spikes, and how many time steps to consider for end of trial analysis
#-------------------------
TimeStepForRecording = 0.05 * second # record responses in statemonitor at this time step
NumTimeStepsToAnalyze = 1

#-------------------------
# equations...
# g == total synaptic input == current synaptic activation + bias + stimulus
# S_rate is rate of sensory neuron which reflects input strength 'g' passed through non-linear transfer function 
#-------------------------
S_eqs = '''
dS_act/dt = -S_act/S_tau : 1
Stim : 1
g = S_act + S_bias + Stim : 1  
S_rate = 0.4*(1+tanh(S_tf_slope*g-3))/S_tau : Hz
'''

#-------------------------
# then make similar equations for the random network
#-------------------------
R_eqs = '''
dR_act/dt = -R_act/R_tau : 1
Stim : 1
g = R_act + R_bias + Stim : 1  
R_rate = 0.4*(1+tanh(R_tf_slope*g-3))/R_tau : Hz
'''

#-------------------------
# baseline input for sensory network
#-------------------------
BaselineScale = 0

#-------------------------
# stimulus strength and precision
#-------------------------
# StimStrength = 10  # set in code below
kappa = 14         # BB2019 used a gaussian...i am using a von Mises and kappa == 14 approximates the std of their gaussian
StimToZero = 3     # stim input goes to 0 <> StimToZero stds from mu 

#-------------------------
# pick rand stim values or pick from 
# S_N_num_stimvals discrete values?
# use for conditional decoding in rand network
#-------------------------
StimRandOrFixed = 1    # if StimRandOrFixed == 1, then pick from from a discrete set of stim values, otherwise random picks
# S_N_num_stimvals = 1  # how many discrete values to pick from?                
# S_stim_pools = [3]   #which pools to present stims too...set below

#-------------------------
# stim exposure time (in seconds)
#-------------------------
StimExposeTime = 0.1 * second  

#-------------------------
# WM delay period (in seconds) - stim goes to baseline during this interval...
#-------------------------
AttnTime = 1.0 * second

#------------------------------------------------ 
# synaptic params for recurrent connections in sensory pools, hence these all have the 'S_' prefix
#------------------------------------------------ 
S_rec_w_baseline = 0.28  # baseline of weight matrix for sensory pools
S_rec_w_amp_exc = 2      # amp of excitatory connections 
S_rec_w_kappa_exc = 1    # dispersion of excitatory connections
S_rec_w_amp_inh = 2      # amp of inhibitory connections
S_rec_w_kappa_inh = .25 #.25  # dispersion of inhibitory connections
S_Excitation = False     #  self excitation? True/False (False default). If False, then zero out main diag of w mat

#------------------------------------------------ 
# synaptic params for recurrent connections in random layer, hence these all have the 'R_' prefix
#------------------------------------------------ 
R_inh_connections = 0

if R_inh_connections:
    R_rec_w_baseline = .28
    R_rec_w_amp_exc = 0 
    R_rec_w_kappa_exc = 0
    R_rec_w_amp_inh = 2
    R_rec_w_kappa_inh = 10
else:
    R_rec_w_baseline = 0
    R_rec_w_amp_exc = 0 
    R_rec_w_kappa_exc = 0
    R_rec_w_amp_inh = 0
    R_rec_w_kappa_inh = 0
    
R_Excitation = False     #  self excitation? True/False (False default). If False, then zero out main diag of w mat

#------------------------------------------------ 
# if laterally connecting the sensory pools - only excitatory
# with amp and prob specified here. These connections will 
# only be between corresponding neurons in each pool 
# (i.e. S.connect(j='i') and vice-versa)
#------------------------------------------------ 
# S_connect_pools = 0
S_pools_to_S_pools_w_amp = 0
S_pools_to_S_pools_proportion = 1

#------------------------------------------------ 
# synaptic params for sensory-to-random layer connections
# Note that the order of the prefix indicates the order of the
# connection - to S_to_R means sensory to random, and R_to_S means
# random to sensory
#------------------------------------------------ 
S_to_R_target_w = 2.1    # feedforward weight before balancing (alpha param from equations in BB2019)
S_to_R_ei_balance = -1    # -1 is perfectly balanced feedforward E/I ratio
S_to_R_baseline = 0       # 

R_to_S_target_w = 0.2    # feedback weight before balancing (beta from equations in BB2019)
R_to_S_ei_balance = -1   # -1 is perfectly balanced feedback E/I ratio
R_to_S_baseline = 0

exc_con_prob = 0.35       # this is gamma from BB2019, or the prob of excitatory connections

weight_factor = 1000.0    # factor for computing weights - make float here (and double check that in class def to enforce)

#-------------------------
# Run a simulation - return the decoded angle/abs(magnitude) 
# and the actual stim values
# pred_ang is a trial by set size by time matrix, as is pred_abs (the magnitude of the estimate)
# stim_ang is a trial by S_N_pools, and stim_pools records which of the S_N_pools got a stimulus on the 
# current trial, so it is a trial by setsize matrix
#------------
rand_top_down = 0
r_stim_ratios = [0.1, 0.2, 0.3, 1] 
# R_StimProportion = 1
# stim_strengths = [1,3,5,7,9] #np.arange(6,10.5)
stim_strengths = np.arange(5,10,.5)
# stim_strengths = [10]
SF_exp = 6            # specificity of top down feedback. 
N_trials = 50

set_sizes = [1]
r_stim_amps = [0,1,2,3,5,7,10]
connect_pools = [0]   # connect sensory pools (1) or not (0)
force_run = 0

S_tf_slopes = [0.25]

crf = np.zeros((len(r_stim_amps),len(stim_strengths)))
mem_resp = np.zeros((len(r_stim_amps),len(stim_strengths)))

S_N_num_stimvals = 1  # how many discrete values to pick from?                
S_stim_pools = [3]   # which pools to present stims too...

#%%
print('Stim Strength levels:',len(stim_strengths))
t = time.time()
iCounter = 0
for R_StimProportion in r_stim_ratios:
    for ss_cnt, StimStrength in enumerate(stim_strengths):
        for SetSize in set_sizes:
            if SetSize==1:
                S_N_num_stimvals = 1  # how many discrete values to pick from?                
                S_stim_pools = [3]   # which pools to present stims too...
            # elif SetSize==2: 
            #     S_N_num_stimvals = 4  # how many discrete values to pick from?                
            #     S_stim_pools = [3,4]   #which pools to present stims too...
            # elif SetSize==4:
            #     S_N_num_stimvals = 2  # how many discrete values to pick from?                
            #     S_stim_pools = [2,3,4,5]   #which pools to present stims too...                
            
            
            for S_connect_pools in connect_pools:
                for r_cnt, R_stim in enumerate(r_stim_amps):
                    for S_tf_slope in S_tf_slopes:
                        #-------------------------
                        # init params for the model
                        #-------------------------
                        init_params = {'rndSeed' : rndSeed, 'N_trials' : N_trials, 'S_N_neuron' : S_N_neuron, 'S_N_pools' : S_N_pools, 
                                      'R_N_neuron' : R_N_neuron, 'S_connect_pools' : S_connect_pools,
                                      'StimRandOrFixed' : StimRandOrFixed, 'S_N_num_stimvals' : S_N_num_stimvals, 'S_stim_pools' : S_stim_pools,
                                      'S_eqs' : S_eqs, 'R_eqs' : R_eqs, 'S_tau' : S_tau, 'R_tau' : R_tau,
                                      'S_bias' : S_bias, 'R_bias' : R_bias, 'S_InitSynRange' : S_InitSynRange,
                                      'R_InitSynRange' : R_InitSynRange,
                                      'TimeStepForRecording' : TimeStepForRecording, 'NumTimeStepsToAnalyze' : NumTimeStepsToAnalyze,
                                      'StimExposeTime' : StimExposeTime, 'AttnTime' : AttnTime,
                                      'S_tf_slope' : S_tf_slope, 'R_tf_slope' : R_tf_slope, 'BaselineScale' : BaselineScale,
                                      'R_StimProportion' : R_StimProportion, 'R_stim' : R_stim, 'SF_exp' : SF_exp, 'StimStrength' : StimStrength, 'kappa' : kappa, 'SetSize' : SetSize,
                                      'StimToZero' : StimToZero, 'N_trials' : N_trials, 
                                      'S_rec_w_baseline' : S_rec_w_baseline, 'S_rec_w_amp_exc' : S_rec_w_amp_exc, 
                                      'S_rec_w_kappa_exc' : S_rec_w_kappa_exc, 'S_rec_w_amp_inh' : S_rec_w_amp_inh, 
                                      'S_rec_w_kappa_inh' : S_rec_w_kappa_inh, 'S_Excitation' : S_Excitation,
                                      'R_rec_w_baseline' : R_rec_w_baseline, 'R_rec_w_amp_exc' : R_rec_w_amp_exc, 
                                      'R_rec_w_kappa_exc' : R_rec_w_kappa_exc, 'R_rec_w_amp_inh' : R_rec_w_amp_inh, 
                                      'R_rec_w_kappa_inh' : R_rec_w_kappa_inh, 'R_Excitation' : R_Excitation,
                                      'S_to_R_target_w' : S_to_R_target_w, 'S_to_R_ei_balance' : S_to_R_ei_balance, 
                                      'S_to_R_baseline' : S_to_R_baseline, 'R_to_S_target_w' : R_to_S_target_w, 
                                      'R_to_S_ei_balance' : R_to_S_ei_balance, 'R_to_S_baseline' : R_to_S_baseline,
                                      'exc_con_prob' : exc_con_prob, 'weight_factor' : weight_factor, 
                                      'S_pools_to_S_pools_w_amp' : S_pools_to_S_pools_w_amp, 
                                      'S_pools_to_S_pools_proportion' : S_pools_to_S_pools_proportion, 'rand_top_down': rand_top_down, 'rndSeed' : rndSeed}
                        
                        
                        
                        #------------------------
                        # Make a file name for output
                        
                        #------------------------
                        # if rand_top_down: 
                        # fn_out = 'rand_attn_data/' + 'Attn-Classify-RandStim-' + str(R_stim) + '_stim-' + str(StimStrength) + '_sSlope-' + str(S_tf_slope) + '_rSlope-' + str(R_tf_slope) + '.npz'
                        fn_out = 'rand_attn_data/' + 'Attn-Classify-Top' + str(R_StimProportion) + 'Stim-' + str(R_stim) + '_stim-' + str(StimStrength) + '_sSlope-' + str(S_tf_slope) + '_rSlope-' + str(R_tf_slope) + '.npz'
                        # else: 
                        #     fn_out = 'rand_attn_data/' + 'Attn-Classify-NoRandStim-RandInh-' + str(R_stim) + '_stim-' + str(StimStrength) + '_ss-' + str(SetSize) + '_connect_pools-' + str(S_connect_pools) + '_r_stim_proportion-' + str(R_StimProportion) + '_num_stimvals-' + str(S_N_num_stimvals) + '.npz'
                        # fn_out = 'tmp'
                        #------------------------
                        # If this sim hasn't already been run, then run it
                        #------------------------           
                        if (path.exists(fn_out) == False) or (force_run == 1):
                            iCounter = iCounter+1
                            print('('+str(iCounter)+'/'+str(np.size(r_stim_ratios)*np.size(stim_strengths)*np.size(r_stim_amps)*np.size(S_tf_slopes))+') Running sim: ' + fn_out)
        
                            #-------------------------
                            # init the model object
                            #-------------------------
                            M = rand_attn_inh(init_params)
        
                            #-------------------------
                            # Run the sims
                            #-------------------------                    
                            pred_ang, pred_abs, mem_abs, stim_ang, stim_pools, S_fr, R_fr, temp = M.run_sim_rand()
                            # print(mem_abs[0,S_stim_pools,-NumTimeStepsToAnalyze])
                        
                            # S_fr[nTrials,nNeurons,timewindow] -- currently averaging over all trials and then last five time points
                            # crf[r_cnt, ss_cnt] = np.mean(np.mean(S_fr[:,S_stim_pools*S_N_neuron,-5:],axis=0))
                            mem_resp[r_cnt, ss_cnt] = np.mean(np.mean(mem_abs[:,S_stim_pools,-5:],axis=0))
                    
                            #-------------------------
                            # save data
                            #-------------------------
                            np.savez(fn_out, S_fr = S_fr, R_fr = R_fr, pred_ang=pred_ang, pred_abs=pred_abs, mem_abs=mem_abs, stim_ang=stim_ang, stim_pools=stim_pools, 
                                    S_N_neuron=S_N_neuron, params=init_params, fp = 'float32', nTimeSteps=int((StimExposeTime+AttnTime)/TimeStepForRecording))
                            
                            #plt.plot(np.mean(S_fr, axis=0)[3*512,:])
                        
                        else: #if path.exists(fn_out) == True:
                            print('Skipping:' + fn_out)
elapsed = time.time() - t
print(elapsed,'sec Elapsed')

#%% Load and plot abs ratio between stimulated and unstimulated pools
# stim_strengths = np.arange(5,10,.5)
# r_stim_amps = [0,1,2,3]
# r_stim_ratios = [0.1, 0.2, 0.3] 
# r_stim_ratios=[1]
S_all_pools = np.arange(S_N_pools)
S_unstim_pools = np.delete(S_all_pools,S_all_pools==S_stim_pools)

mem_resp = np.zeros((len(r_stim_amps),len(stim_strengths)))

fig, axs = plt.subplots(2, len(r_stim_ratios),figsize=(10,5))
# axs[0, 0].plot(x, y)
# axs[0, 0].set_title('Axis [0, 0]')
# axs[0, 1].plot(x, y, 'tab:orange')
# axs[0, 1].set_title('Axis [0, 1]')
# axs[1, 0].plot(x, -y, 'tab:green')
# axs[1, 0].set_title('Axis [1, 0]')
# axs[1, 1].plot(x, -y, 'tab:red')
# axs[1, 1].set_title('Axis [1, 1]')


    
for r_prop_cnt, R_StimProportion in enumerate(r_stim_ratios):
    for S_tf_slope in S_tf_slopes:
        for r_cnt, R_stim in enumerate(r_stim_amps):
            for ss_cnt, StimStrength in enumerate(stim_strengths):
            
                
                # load saved data
                # fn_out = 'rand_attn_data/' + 'Attn-Classify-RandStim-' + str(R_stim) + '_stim-' + str(StimStrength) + '_sSlope-' + str(S_tf_slope) + '_rSlope-' + str(R_tf_slope) + '.npz'
                fn_out = 'rand_attn_data/' + 'Attn-Classify-Top' + str(R_StimProportion) + 'Stim-' + str(R_stim) + '_stim-' + str(StimStrength) + '_sSlope-' + str(S_tf_slope) + '_rSlope-' + str(R_tf_slope) + '.npz'
                # fn_out = 'rand_attn_data/' + 'Attn-Classify-Top' + str(R_StimProportion) + 'Stim-' + str(R_stim) + '_stim-' + str(StimStrength) + '_sSlope-' + str(S_tf_slope) + '_rSlope-' + str(R_tf_slope) + '.npz'
                data = np.load(fn_out)
                # # grab mem_abs
                # mem_abs = data['mem_abs']
                # # average and store for each combination of bottom-up x top-down stim strength
                # mem_resp[r_cnt, ss_cnt] = np.mean(np.mean(mem_abs[:,S_stim_pools,-5:],axis=0))
                
                # grab pred_abs
                mem_abs = data['pred_abs']
                # average and store for each combination of bottom-up x top-down stim strength
                stim_abs = np.mean(np.mean(mem_abs[:,S_stim_pools,-5:],axis=0))
                unstim_abs = np.mean(np.mean(np.mean(mem_abs[:,S_unstim_pools,-5:],axis=0),axis=1))
                mem_resp[r_cnt, ss_cnt] = stim_abs/unstim_abs
            
    # for r_cnt, R_stim in enumerate(r_stim_amps):
            # plt.plot(stim_strengths,mem_resp[r_cnt,])
            axs[0, r_prop_cnt].plot(stim_strengths,mem_resp[r_cnt,])
    
        # plt.xlabel('Stimulus Strength')
        # plt.ylabel('stim_abs/unstim_abs')
        axs[0, r_prop_cnt].set_title('rProp: '+str(R_StimProportion))
        # plt.title('sSlope: '+str(S_tf_slope)+' rSlope: '+str(R_tf_slope)+' rProp: '+str(R_StimProportion))
        if r_prop_cnt == 0:
            axs[0, r_prop_cnt].legend(r_stim_amps)
            axs[0, r_prop_cnt].set(ylabel='stim_abs/unstim_abs')
        # plt.show()
fig.suptitle('sSlope: '+str(S_tf_slope)+' rSlope: '+str(R_tf_slope))
fig.tight_layout()
# for ax in axs.flat:
#     ax.set(xlabel='x-label', ylabel='y-label')

# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()

#%% Load and plot CRF
# stim_strengths = np.arange(5,10,.5)
# r_stim_amps = [0,1,2,3]

S_all_pools = np.arange(S_N_pools)
S_unstim_pools = np.delete(S_all_pools,S_all_pools==S_stim_pools)

mem_resp = np.zeros((len(r_stim_amps),len(stim_strengths)))
for R_StimProportion in r_stim_ratios:
    for S_tf_slope in S_tf_slopes:
        for r_cnt, R_stim in enumerate(r_stim_amps):
            for ss_cnt, StimStrength in enumerate(stim_strengths):
        
            
                # load saved data
                # fn_out = 'rand_attn_data/' + 'Attn-Classify-RandStim-' + str(R_stim) + '_stim-' + str(StimStrength) + '_sSlope-' + str(S_tf_slope) + '_rSlope-' + str(R_tf_slope) + '.npz'
                fn_out = 'rand_attn_data/' + 'Attn-Classify-Top' + str(R_StimProportion) + 'Stim-' + str(R_stim) + '_stim-' + str(StimStrength) + '_sSlope-' + str(S_tf_slope) + '_rSlope-' + str(R_tf_slope) + '.npz'
                data = np.load(fn_out)
                # # grab mem_abs
                mem_abs = data['mem_abs']
                # # average and store for each combination of bottom-up x top-down stim strength
                mem_resp[r_cnt, ss_cnt] = np.mean(np.mean(mem_abs[:,S_stim_pools,-5:],axis=0))
            
        # for r_cnt, R_stim in enumerate(r_stim_amps):
            # plt.plot(stim_strengths,mem_resp[r_cnt,])
            axs[1, r_prop_cnt].plot(stim_strengths,mem_resp[r_cnt,])
    
        # plt.xlabel('Stimulus Strength')
        # plt.ylabel('Decoding in Stimulated Pool')
        # plt.title('Slope: '+str(S_tf_slope))
        # plt.title('sSlope: '+str(S_tf_slope)+' rSlope: '+str(R_tf_slope)+' rProp: '+str(R_StimProportion))
        axs[1, r_prop_cnt].set_title('rProp: '+str(R_StimProportion))
        if r_prop_cnt == 0:
            axs[1, r_prop_cnt].set(ylabel='Decoding in Stimulated Pool')
        
        # plt.legend(r_stim_amps)
        # plt.show()
for ax in axs.flat:
    ax.set(xlabel='Stimulus Strength', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
#%% see what's happening in unstimulated pools
S_all_pools = np.arange(S_N_pools)
S_unstim_pools = np.delete(S_all_pools,S_all_pools==S_stim_pools)

stim_strengths = np.arange(5,10,.25)
stim_strengths = [7.5]
N_trials = 50
r_stim_amps = [3]
for ss_cnt, StimStrength in enumerate(stim_strengths):
    for r_cnt, R_stim in enumerate(r_stim_amps):
        fn_out = 'rand_attn_data/' + 'Attn-Classify-RandStim-' + str(R_stim) + '_stim-' + str(StimStrength) + '.npz'
        data = np.load(fn_out)
pred_ang = data['pred_ang'] # trial x sensory pools x timepoint
pred_abs = data['pred_abs'] # trial x sensory pools x timepoint
# self.plt_raster(S_spike, 'Sensory') # only run if trial is one
# self.plt_raster(R_spike, 'Random')

#%%
for p_cnt, S_pool in enumerate(S_all_pools):
    plt.plot(np.arange(shape(pred_abs)[2]),pred_abs[5,S_pool,])
plt.xlabel('Time')
plt.ylabel('Response')
plt.title('S stim:'+str(StimStrength)+'  R stim:'+str(R_stim))
plt.legend(S_all_pools)
plt.show()

plt.savefig('SensoryResponse_' + 'S'+str(StimStrength)+'_R'+str(R_stim)+'.jpg', dpi=600)
                                