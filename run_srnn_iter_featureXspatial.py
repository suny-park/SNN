# for VSS2023
# Present two stimuli to the first pool, no stimulus presented to other pools
# Apply top-down gain to one of the two stimulus, 
# by gain to random neurons preferring that stimulus
# Decode attended stimulus.

# Implemented localizer task
# 1: Initialize network and connections
# 2: Present one out of 16 equally spaced stimuli (in all pools or one pool at a time?) 
# - to record responses to each stimulus and use for decoding
# RESET
# 3: 
# need to do this before importing brian2
# thisrep = input("Which rep? ")
# thisrep = int(thisrep)

thiskappa = input("Which kappa? ")
thiskappa = float(thiskappa)

from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.stats import circmean
import os.path
from os import path
# import brian2genn
import numpy.matlib


# os.chdir("C:\Users\boss\OneDrive - UC San Diego\Lab\RNN project\forVSS\result_figure")
# import SVC classifier
from sklearn.svm import SVC
# import metrics to compute accuracy
from sklearn.metrics import accuracy_score
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

from circ_reg_fromJohn import *
from circ_corr import *


#%%

# need this to use genn
# set_device('genn')

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
StimExposeTime = 0.3 * second  
# StimExposeTime = 1.0 * second  

#-------------------------
# WM delay period (in seconds) - stim goes to baseline during this interval...
#-------------------------
AttnTime = 1.0 * second
# AttnTime = 0.3 * second

#------------------------------------------------ 
# synaptic params for recurrent connections in sensory pools, hence these all have the 'S_' prefix
#------------------------------------------------ 
S_rec_w_baseline = 0.28  # baseline of weight matrix for sensory pools
S_rec_w_amp_exc = 2      # amp of excitatory connections 
S_rec_w_kappa_exc = 3    # dispersion of excitatory connections
S_rec_w_amp_inh = 2      # amp of inhibitory connections
S_rec_w_kappa_inh = .83  # dispersion of inhibitory connections, higher is less dispersion, less interference between stimuli -- .83 - 8/28/24
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
S_pools_to_S_pools_w_amp = 1
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

#%%-------------------------
# Run a simulation
#-------------------------
doPlot = 0 # plot stuff as we go - recommended for a first run
saveFile = 1 # save file or not
S_connect_pools = 0 # connect between first layer sub-networks

s_stim_amps = np.arange(20) # strength of bottom-up stimulus
s_stim_amps = [10]
r_stim_amps_F = [0,2,4,6,8] # strength of top-down gain for feature attention
r_stim_amps_S = [0,2,4,6,8] # strength of top-down gain for spatial attention
r_stim_amps_S = [0]

r_stim_ratios = [0.2] # proportion of second layer units to apply top-down gain
r_conn_kappas = [0] # 0 for now
# r_conn_kappas = [0,0.1,0.2,0.3,0.4] # spread of between-layer connections. higher value means more structure

N_trials_sensory = 0 # separate values for sensory and attention task trials
N_trials_attention = 50 #
N_stim_loc = 16 # number of stimuli in the sensory task
N_stim_main = 2 # number of stimuli in the attention task


# s_stim_amps = [7,10,15] # strength of bottom-up stimulus
# r_stim_amps_F = [6,8] # strength of top-down gain for feature attention
# r_stim_amps_S = [6,8] # strength of top-down gain for spatial attention
# N_trials_attention = 1 #

from datetime import datetime
now = datetime.now()
# dd/mm/YY H:M:S
date_string = now.strftime("%m%d%Y_%H%M%S")
    
#%%
from srnn_iter_featureXspatial import * # Make sure this is the one we want!!

N_reps = 1 # run one initialization at a time to run on multiple terminals
shiftStep = 32
shiftReps = 1 #16

# for rand_kappa in r_conn_kappas:
rand_kappa = thiskappa
rep = 1
# rep = thisrep
# for rep in np.arange(N_reps):
# for rep in reps:
t = time.time()
print('running rep '+str(rep))
#-------------------------
# Set rnd seed so that can repeat model...important! deterministic from here on out...
#-------------------------
rndSeed = rep

#-------------------------
# init params for the model
#-------------------------
init_params = {'rndSeed' : rndSeed, 'N_trials_sensory' : N_trials_sensory, 'N_trials_attention' : N_trials_attention, 
              'S_N_neuron' : S_N_neuron, 'S_N_pools' : S_N_pools, 
              'R_N_neuron' : R_N_neuron, 'S_connect_pools' : S_connect_pools,
              'StimRandOrFixed' : StimRandOrFixed,
              'S_eqs' : S_eqs, 'R_eqs' : R_eqs, 'S_tau' : S_tau, 'R_tau' : R_tau,
              'S_bias' : S_bias, 'R_bias' : R_bias, 'S_InitSynRange' : S_InitSynRange,
              'R_InitSynRange' : R_InitSynRange,
              'TimeStepForRecording' : TimeStepForRecording, 'NumTimeStepsToAnalyze' : NumTimeStepsToAnalyze,
              'StimExposeTime' : StimExposeTime, 'AttnTime' : AttnTime,
              'S_tf_slope' : S_tf_slope, 'R_tf_slope' : R_tf_slope, 'BaselineScale' : BaselineScale,
              'kappa' : kappa,
              'StimToZero' : StimToZero,
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
              'S_pools_to_S_pools_proportion' : S_pools_to_S_pools_proportion, 
              'doPlot' : doPlot,
              'N_stim_loc': N_stim_loc,'N_stim_main' : N_stim_main,
              's_stim_amps' : s_stim_amps, 'r_stim_amps_F' : r_stim_amps_F, 'r_stim_amps_S' : r_stim_amps_S, 'r_stim_ratios' : r_stim_ratios,
              'shiftStep':shiftStep, 'shiftReps':shiftReps,'rand_kappa':rand_kappa}

#-------------------------
# init the model object
#-------------------------
M = rand_attn_inh(init_params)

#-------------------------
# Run the sims
#-------------------------                    
S_fr_avg_loc, label_stim_loc, label_pool_loc, label_trial_loc, \
    S_fr_avg_main, label_stim_main, label_trial_main, \
        fr_angle, fr_abs, fr_att_abs, label_attPool_main, label_stim_strength_main = M.run_sim_rand()
#fn_decoding = 'sim_decoding/FBA_decoding_Rep0.npz'
#fn_decoding = 'FBA_decoding_shift_Rep'+str(rep)+'.npz'
fn_decoding = 'results/FxS_kappa-'+str(rand_kappa)+'_seed-'+str(rep)+'_test.npz'
if saveFile:
    np.savez(fn_decoding, S_fr_avg_loc=S_fr_avg_loc, label_stim_loc=label_stim_loc, \
             label_pool_loc=label_pool_loc, label_trial_loc=label_trial_loc, \
                 S_fr_avg_main= S_fr_avg_main, label_stim_main= label_stim_main, \
                     label_trial_main=label_trial_main, N_stim_loc=N_stim_loc, \
                         r_stim_amps_F=r_stim_amps_F, r_stim_amps_S=r_stim_amps_S, r_stim_ratios=r_stim_ratios, s_stim_amps=s_stim_amps, \
                                 N_trials_sensory=N_trials_sensory,N_trials_attention=N_trials_attention,S_N_pools=S_N_pools,S_N_neuron=S_N_neuron, \
                                     fr_angle=fr_angle, fr_abs=fr_abs, fr_att_abs=fr_att_abs, label_attPool_main=label_attPool_main, label_stim_strength_main=label_stim_strength_main)
    print('Saved '+fn_decoding)
                        
elapsed = time.time() - t
print(round(elapsed/60),'min Elapsed')
print()
        
        # print(fr_att_abs[1,:,-1,:,0])
       















