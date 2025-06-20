# Random connections support precise top-down modulations of early sensory cortex

Scripts used to simulate top-down feature-based attention in spiking neural network models.

Adapted from Bouchacourt & Buschman (2019): https://www.cell.com/neuron/fulltext/S0896-6273(19)30377-0. Original code: https://github.com/buschman-lab/FlexibleWorkingMemory

Present two stimuli to the first sub-network (no stimulus presented to others)

Apply top-down modulation to second layer for one of the two stimulus.


**run_srnn_iter_feature.py**

A wrapper script that can change network and task properties. This script calls functions in `srnn_iter_feature.py` that actually execute the simulations.

Key properties:
- stim_strengths: strength of bottom-up stimulus input
- r_stim_amps: strength of top-down modulation
- r_stim_ratios: proportion of second layer units to apply top-down modulation - fixed to .2 for main simulations
- r_conn_kappas: spread of between-layer connections. Higher value means more structure.



**srnn_iter_feature.py**

Script containing functions needed for the simulations. 

Runs Sensory Task which presents one stimulus to one sub-network at a time, and Attention Task which presents two stimuli to one sub-network and applies top-down attentional modulation for one of the two stimuli to the second layer neurons with preference to that target stimulus.

Saves out `.npz` files that contain firing rates of first and second layer neurons across time and trial information such as stimulus strength, top-down modulation strength, attended stimulus, and etc. 


**analyze_feature.py**

Analyzes `.npz` files from `run_srnn_iter_feature.py` and saves summarized `.mat` files.

Runs these analyses:
1. Train/test circular regression model on sensory task to predict presented stimulus.
2. Train circular regression model on sensory task, test on attention task to predict attended stimulus (Figure 3A,B).
3. Train/test SVM on attention task within each sub-network (Figure 3C).
4. Train SVM on attention task for one unstimulated sub-network then test on another unstimulated sub-network (Figure 3D).

Then calculates average firing rates to plot firing rates for 
- attended/unattended stimulus in the stimulated sub-network (Figure 2C,D,E)
- attended stimulus in the stimulated/unstimulated sub-network (Figure 5)


**plotSNN.m**

Plots all main figures using `.mat` files from `analyze_feature.py`

