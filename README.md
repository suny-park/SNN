# Random connections support precise top-down modulations of early sensory cortex

Scripts used to simulate top-down feature-based attention in spiking neural network models.
Adapted from Bouchacourt & Buschman (2019): https://www.cell.com/neuron/fulltext/S0896-6273(19)30377-0
Original code: https://github.com/buschman-lab/FlexibleWorkingMemory
Present two stimuli to the first sub-netwok (no stimulus presented to others)
Apply top-down modulation to second layer for one of the two stimulus.


**run_srnn_iter_feature.py**

A wrapper script that can change network and task properties. This script calls functions in `srnn_iter_feature.py` that actually execute the simulations.


**srnn_iter_feature.py**

Script containing functions needed for the simulations. 
Runs Sensory Task which presents one stimulus to one sub-network at a time, and Attention Task which presents two stimuli to one sub-network and applies top-down attentional modulation for one of the two stimuli to the second layer neurons with preference to that target stimulus.
Saves out .npz files that contain firing rates of first and second layer neurons across time and trial information such as stimulus strength, top-down modulation strength, attended stimulus, and etc. 


**analyze_feature.py**

Analyzes output files from `run_srnn_iter_feature.py` and saves summarized `.mat` files.


**plotSNN.m**

Plots figures.
