# TODO: 
# add task overview at the top
# check what's used from numpy.matlib?

# Feature-based Attention Simulation in a Spiking Neural Network Model
# Present two stimuli to the first sub-netwok (no stimulus presented to others)
# Apply top-down gain to second layer for one of the two stimulus.

# Sensory Task
# 1: Initialize network and connections
# 2: Present one out of 16 equally spaced stimuli (in all pools or one pool at a time?) 
# - to record responses to each stimulus and use for decoding
# 3: RESET

# Attention Task


from brian2 import *
import numpy as np
from numpy import pi
import os.path
from os import path

from srnn_iter_feature import * # Make sure this is the one we want!!

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
# 8/4/22 slopes fixed at .25 

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
kappa = 14         # for von Mises
StimToZero = 3     # stim input goes to 0 <> StimToZero stds from mu 

#-------------------------
# stim exposure time (in seconds)
#-------------------------
StimExposeTime = 0.3 * second  

#-------------------------
# Attention Task stimulus time (in seconds)
#-------------------------
AttnTime = 1.0 * second

#------------------------------------------------ 
# synaptic params for recurrent connections in sensory pools, hence these all have the 'S_' prefix
#------------------------------------------------ 
S_rec_w_baseline = 0.28  # baseline of weight matrix for sensory pools
S_rec_w_amp_exc = 2      # amp of excitatory connections 
S_rec_w_kappa_exc = 3    # dispersion of excitatory connections
S_rec_w_amp_inh = 2      # amp of inhibitory connections
S_rec_w_kappa_inh = .83  # dispersion of inhibitory connections, higher is less dispersion, less interference between stimuli -- .83 - 8/28/24
S_SelfExcitation = False     #  self excitation? True/False (False default). If False, then zero out main diag of w mat

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

R_SelfExcitation = False     #  self excitation? True/False (False default). If False, then zero out main diag of w mat

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

exc_con_prob = 0.35       # prob of excitatory connections

weight_factor = 1000.0    # factor for computing weights - make float here (and double check that in class def to enforce)

#-------------------------
# Run a simulation
#-------------------------
doPlot = 1 # plot stuff as we go - recommended for a first run
saveFile = 1 # save file or not
saveSpikeDat = 0 # save spike data for drawing raster plots offline

stim_strengths = np.arange(0,20) # strength of bottom-up stimulus
r_stim_amps = np.arange(0,20,2) # strength of top-down gain
r_stim_ratios = [0.2] # proportion of second layer units to apply top-down gain - fixed to .2
r_conn_kappas = [0,0.1,0.2,0.3,0.4] # spread of between-layer connections. higher value means more structure

N_trials_sensory = 100 # number of trials for the sensory task
N_trials_attention = 50 # number of trials for the attention task

N_stim_loc = 16 # number of stimuli in the sensory task
N_stim_main = 2 # number of stimuli in the attention task

shiftStep = 32 # number of steps to shift the stimulus pair in the attention task
shiftReps = 1 # used to be 16 - don't shift and keep attStimShift at 8

for rand_kappa in r_conn_kappas:

    rep=1
    t = time.time()

    #-------------------------
    # Set rnd seed so that can repeat model...important! deterministic from here on out...
    #-------------------------
    rndSeed = rep

    #-------------------------
    # init params for the model
    #-------------------------
    init_params = {'rndSeed' : rndSeed, 'N_trials_sensory' : N_trials_sensory, 'N_trials_attention' : N_trials_attention, 
                  'S_N_neuron' : S_N_neuron, 'S_N_pools' : S_N_pools, 'R_N_neuron' : R_N_neuron, 
                  'S_eqs' : S_eqs, 'R_eqs' : R_eqs, 'S_tau' : S_tau, 'R_tau' : R_tau,
                  'S_bias' : S_bias, 'R_bias' : R_bias, 
                  'S_InitSynRange' : S_InitSynRange,'R_InitSynRange' : R_InitSynRange,
                  'TimeStepForRecording' : TimeStepForRecording, 'NumTimeStepsToAnalyze' : NumTimeStepsToAnalyze,
                  'StimExposeTime' : StimExposeTime, 'AttnTime' : AttnTime,
                  'S_tf_slope' : S_tf_slope, 'R_tf_slope' : R_tf_slope, 'BaselineScale' : BaselineScale,
                  'kappa' : kappa, 'StimToZero' : StimToZero,
                  'S_rec_w_baseline' : S_rec_w_baseline, 'S_rec_w_amp_exc' : S_rec_w_amp_exc, 
                  'S_rec_w_kappa_exc' : S_rec_w_kappa_exc, 'S_rec_w_amp_inh' : S_rec_w_amp_inh, 
                  'S_rec_w_kappa_inh' : S_rec_w_kappa_inh, 'S_SelfExcitation' : S_SelfExcitation,
                  'R_rec_w_baseline' : R_rec_w_baseline, 'R_rec_w_amp_exc' : R_rec_w_amp_exc, 
                  'R_rec_w_kappa_exc' : R_rec_w_kappa_exc, 'R_rec_w_amp_inh' : R_rec_w_amp_inh, 
                  'R_rec_w_kappa_inh' : R_rec_w_kappa_inh, 'R_SelfExcitation' : R_SelfExcitation,
                  'S_to_R_target_w' : S_to_R_target_w, 'S_to_R_ei_balance' : S_to_R_ei_balance, 
                  'S_to_R_baseline' : S_to_R_baseline, 'R_to_S_target_w' : R_to_S_target_w, 
                  'R_to_S_ei_balance' : R_to_S_ei_balance, 'R_to_S_baseline' : R_to_S_baseline,
                  'exc_con_prob' : exc_con_prob, 'weight_factor' : weight_factor, 
                  'doPlot' : doPlot,
                  'N_stim_loc': N_stim_loc,'N_stim_main' : N_stim_main,
                  'stim_strengths' : stim_strengths, 'r_stim_amps' : r_stim_amps, 'r_stim_ratios' : r_stim_ratios,
                  'shiftStep':shiftStep, 'shiftReps':shiftReps,'rand_kappa':rand_kappa,
                  'saveSpikeDat':saveSpikeDat}

    #-------------------------
    # init the model object
    #-------------------------
    M = rand_attn_inh(init_params)

    #-------------------------
    # Run the sims
    #-------------------------                    
    S_fr_avg_loc, label_stim_loc, label_pool_loc, label_trial_loc, \
        S_fr_avg_main, label_stim_main, label_trial_main, \
            fr_angle, fr_abs, fr_att_abs, label_stim_strength_main = M.run_sim_rand()
    
    fn_decoding = 'results/F_kappa-'+str(rand_kappa)+'_seed-'+str(rep)+'.npz'

    if saveFile:
        np.savez(fn_decoding, S_fr_avg_loc=S_fr_avg_loc, label_stim_loc=label_stim_loc, \
                 label_pool_loc=label_pool_loc, label_trial_loc=label_trial_loc, \
                     S_fr_avg_main= S_fr_avg_main, label_stim_main= label_stim_main, \
                         label_trial_main=label_trial_main, N_stim_loc=N_stim_loc, \
                             r_stim_amps=r_stim_amps, r_stim_ratios=r_stim_ratios, stim_strengths=stim_strengths, \
                                     N_trials_sensory=N_trials_sensory,N_trials_attention=N_trials_attention,S_N_pools=S_N_pools,S_N_neuron=S_N_neuron, \
                                         label_stim_strength_main=label_stim_strength_main)
        print('Saved '+fn_decoding)
                            
    elapsed = time.time() - t
    print(round(elapsed/60),'min Elapsed')
        