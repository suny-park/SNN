# js 02022022, based in part on BB2019 Neuron model

from brian2 import *
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from scipy.special import i0, i1 
import time

# can't get this working on a mac, but works on linux
# if sys.platform != 'darwin':
import cython
prefs.codegen.target = 'numpy' 
prefs.core.default_float_dtype = float32 # try compile to c++

import gc as gcPython

class rand_attn_inh:
    """
    Class to implement spiking network model. First layer is tuned sensory
    neurons that randomly project to 2nd layer
    
    Naming convention:
        S_ starts variables for the sensory layer
        R_ starts varialbes for the random layer
        S/R_N_x is number of x
    """
    
    def __init__(self, params):
        """
        Parameters
        ----------
        params : dictionary of initialization values (or setting defaults)

        Returns
        -------
        None.

        """
        
        #------------------------------------------------ 
        # number of neurons in each layer
        #------------------------------------------------ 
        self.S_N_neuron = params.get('S_N_neuron', 512)
        self.R_N_neuron = params.get('R_N_neuron', 1024)
        
        #------------------------------------------------ 
        # number of sensory pools and whether to connect them
        #------------------------------------------------ 
        self.S_N_pools = params.get('S_N_pools', 1)
        self.S_connect_pools = params.get('S_connect_pools', 0)  # default is 0 (independent pools)

        #------------------------------------------------ 
        # tau
        #------------------------------------------------ 
        self.S_tau = params.get('S_tau', 0.2)
        self.R_tau = params.get('R_tau', 0.2)
        
        #------------------------------------------------ 
        # bias
        #------------------------------------------------ 
        self.S_bias = params.get('S_bias', 0)
        self.R_bias = params.get('R_bias', 0)
        
        #------------------------------------------------ 
        # slope of transfer function
        #------------------------------------------------ 
        self.S_tf_slope = params.get('S_tf_slope', 0.25)
        self.R_tf_slope = params.get('R_tf_slope', 0.25)

        #------------------------------------------------ 
        # initial synapse range
        #------------------------------------------------ 
        self.S_InitSynRange = params.get('S_InitSynRange', 0)
        self.R_InitSynRange = params.get('R_InitSynRange', 0)

        #------------------------------------------------ 
        # time window to save spikes and the expose stim and delay (in seconds)
        #------------------------------------------------ 
        self.TimeStepForRecording = params.get('TimeStepForRecording', .1 * second)
        self.StimExposeTime = params.get('StimExposeTime', .25 * second)
        self.AttnTime = params.get('AttnTime', 1.5 * second)
        
        #------------------------------------------------         
        # equations
        #------------------------------------------------ 
        self.S_eqs = params.get('S_eqs')
        self.R_eqs = params.get('R_eqs')
        
        #------------------------------------------------             
        # stimulus (input) params
        #------------------------------------------------ 
        self.StimExposeTime = params.get('StimExposeTime', .5 * second)
        self.BaselineScale = params.get('BaselineScale', 0)
        self.StimStrength = params.get('StimStrength', 10)
        self.kappa = params.get('kappa', 14)
        self.StimToZero = params.get('StimToZero', 3)
        self.SetSize = params.get('SetSize', 4)                                 # WM set size 
        self.StimRandOrFixed = params.get('StimRandOrFixed', 0)                 # if StimRandOrFixed == 1, then pick from from a discrete set of stim values, otherwise random picks
        self.S_N_num_stimvals = params.get('S_N_num_stimvals', 4)               # if StimRandOrFixed == 1, then pick from this many possible stim values
        self.S_stim_pools = params.get('S_stim_pools', [3])  # what pools to present stims to
        
        #------------------------------------------------ 
        # stim level for random layer
        #------------------------------------------------         
        self.R_stim = params.get('R_stim', 0)                                   # amplitude of stim to random layer
        self.rand_top_down =  params.get('rand_top_down', 0)
        self.R_StimProportion = params.get('R_StimProportion', 0)               # proportion of cells in random layer to stimulate (based on their activity level)
        self.R_NumNeuronsToStim = int(np.floor(self.R_StimProportion * self.R_N_neuron))
        # self.R_NumNeuronsToStim = params.get('R_NumNeuronsToStim', 0)
        self.SF_exp = params.get('SF_exp', 6)                               # spatial selectivity of top down feedback to rand layer.         
        
        #------------------------------------------------ 
        # recurrent synaptic weight stuff for sensory pools
        #------------------------------------------------ 
        self.S_rec_w_baseline = params.get('S_rec_w_baseline', 0.28)    # baseline of weight matrix for sensory pools
        self.S_rec_w_amp_exc = params.get('S_rec_w_amp_exc', 2)         # amp of excitatory connections
        self.S_rec_w_kappa_exc = params.get('S_rec_w_kappa_exc', 1)     # dispersion of excitatory connections
        self.S_rec_w_amp_inh = params.get('S_rec_w_amp_inh', 2)         # amp of inhibitory connections
        self.S_rec_w_kappa_inh = params.get('S_rec_w_kappa_inh', 0.25)  # dispersion of inhibitory connections
        self.S_SelfExcitation = params.get('S_SelfExcitation', False)

        #------------------------------------------------ 
        # recurrent synaptic weight stuff for random layer
        #------------------------------------------------ 
        self.R_rec_w_baseline = params.get('R_rec_w_baseline', 0.28)    # baseline of weight matrix for sensory pools
        self.R_rec_w_amp_exc = params.get('R_rec_w_amp_exc', 0)         # amp of excitatory connections
        self.R_rec_w_kappa_exc = params.get('R_rec_w_kappa_exc', 0)     # dispersion of excitatory connections
        self.R_rec_w_amp_inh = params.get('R_rec_w_amp_inh', 2)         # amp of inhibitory connections
        self.R_rec_w_kappa_inh = params.get('R_rec_w_kappa_inh', 1)  # dispersion of inhibitory connections
        self.R_SelfExcitation = params.get('R_SelfExcitation', False)


        #------------------------------------------------ 
        # synaptic params for between sensory pool connections
        # these are simple reciprocal excitatory connections
        # so don't need as many params
        #------------------------------------------------ 
        self.S_pools_to_S_pools_w_amp = params.get('S_pools_to_S_pools_w_amp', 1)
        self.S_pools_to_S_pools_proportion = params.get('S_pools_to_S_pools_proportion', 1)

        #------------------------------------------------ 
        # synaptic params for sensory-to-random layer connections
        #------------------------------------------------ 
        self.S_to_R_target_w = params.get('S_to_R_target_w', 2.1)       # feedforward weight before balancing (alpha from BB2019)
        self.S_to_R_ei_balance = params.get('S_to_R_ei_balance', -1)    # -1 is perfectly balanced feedforward
        self.S_to_R_baseline = params.get('S_to_R_baseline', 0)
        
        self.R_to_S_target_w = params.get('R_to_S_target_w', 0.2)     # feedback weight before balancing (beta from BB2019)
        self.R_to_S_ei_balance = params.get('R_to_S_ei_balance', -1)    # -1 is perfectly balanced feedback
        self.R_to_S_baseline = params.get('R_to_S_baseline', 0)
        self.exc_con_prob = params.get('exc_con_prob', 0.35)            # this is gamma from BB2019
        self.weight_factor = float(params.get('weight_factor', 1000.0)) # factor for computing weights - make float (have extra typecasting in there just in case)

        #------------------------------------------------ 
        # number of trials
        #------------------------------------------------ 
        self.N_trials = params.get('N_trials', 1)
        
        #------------------------------------------------ 
        # random seed - this is important! will determine
        # all subsequent output!
        #------------------------------------------------ 
        self.rndSeed = params.get('rndSeed', 0)
        np.random.seed(self.rndSeed)
        seed(self.rndSeed)
            
    #------------------------------------------------        
    #------------------------------------------------
    # Start a section of stuff to make baseline + stim input here
    #------------------------------------------------
    #------------------------------------------------

    def baseline_input(self):
        """
        Random baseline input level 


        Returns
        -------
        base_in : numpy array of size = [#pools, #neuron]

        """
        #------------------------------------------------ 
        # define random baseline input for each pool 
        # if you want them to be different...not really used 
        # for anything at the moment and is just zeros (01042022)
        #------------------------------------------------ 
        base_in = np.zeros((self.S_N_pools, self.S_N_neuron))
        for i in np.arange(self.S_N_pools):
            base_in[i,:] = self.BaselineScale * np.random.rand(self.S_N_neuron) 


        #------------------------------------------------ 
        # flatten so can pass without looping over pools
        #------------------------------------------------ 
        return base_in.flatten()

    
    def get_stim_vals(self):
        """
        Generate a set of stimulus values for the current trial

        Returns
        -------
        StimVals for current trial
        StimPools - index into the pools that got a stim on this trial

        """
 
        #------------------------------------------------                
        # nan indicates no stim for a given pool  - doing this because '0'
        # is a valid stim
        #------------------------------------------------        
        StimVals = np.full(self.S_N_pools, np.nan) 
    
        # pick the pools that will get a stim
        StimPools = np.random.choice(self.S_stim_pools, self.SetSize, replace = False)
        
        # if randomly picking a stim from 0:S_N_neuron
        if self.StimRandOrFixed == 0:
            StimVals[StimPools] = np.random.randint(0, self.S_N_neuron, self.SetSize, dtype='int64')

        # pick from one of S_N_num_stimvals values that are evenly spaced over 
        # S_N_neuron
        elif self.StimRandOrFixed == 1:
            poss_stims = np.linspace(0, self.S_N_neuron-(self.S_N_neuron / self.S_N_num_stimvals), self.S_N_num_stimvals).astype(int)
            StimVals[StimPools] = np.random.choice(poss_stims, self.SetSize, replace = True)
            
        return StimVals, StimPools
    
    
    def define_stims(self):
        """
        Define the input stimulus, then scale based on stim strength 
        
        todo: truncate to 0 beyond 'StimToZero' stds
        todo: modify this so that just make the stim function once,
        truncate it, then np.roll it to the right center. Faster and 
        don't have to deal with wrapping issues when truncating
        
        Returns
        -------
        stim : np.array defining input stimulus (or stimuli)

        """        
        
        #------------------------------------------------ 
        # define x (stimulus space) over 2*pi to eval von Mises
        #------------------------------------------------ 
        x = np.linspace(0, (pi * 2)-((pi * 2) / self.S_N_neuron), self.S_N_neuron)

        #------------------------------------------------         
        # compute the circular normal stim function centered on pi
        #------------------------------------------------ 
        stim = np.exp(self.kappa * np.cos(x-pi))/(2 * pi * i0(self.kappa)) 
  
        #------------------------------------------------         
        # get std and figure out cutoff point (truncation points)
        # then apply to stim
        #------------------------------------------------ 
        cutoff = self.StimToZero * np.sqrt(1 - i1(self.kappa)/i0(self.kappa))
        cutoff = len(x[x<=cutoff])   # convert to neuron space
        stim[:int(self.S_N_neuron/2)-cutoff] = 0
        stim[int(self.S_N_neuron/2)+cutoff:] = 0
    
        #------------------------------------------------         
        # normalize to amp == 1
        #------------------------------------------------ 
        stim /= np.max(stim)

        #------------------------------------------------         
        # scale by stim strength
        #------------------------------------------------ 
        stim = self.StimStrength * stim

        return stim
    
    def roll_stims(self, StimVals):
        """
        use define_stims to design the stim shape (and truncate)
        then use this to roll that stim to the desired position 
        in neuron space...this way you only have to set up the 'stim'
        once...

        Returns
        -------
        stims : S_N_pools x S_N_neuron matrix of stims
        rolled to desired position in neuron space (or zeros if no stim
        for a pool)

        """
        
        stims = np.zeros((self.S_N_pools, self.S_N_neuron))
        for sp in np.arange(self.S_N_pools):
            # compute stim drive for the current pool if there is a stim
            # for the pool
            if ~np.isnan(StimVals[sp]): # is a stim is being applied to this pool? 
                #------------------------------------------------         
                # roll stim to be centered at desired location
                #------------------------------------------------ 
                print(StimVals[sp])
                stims[sp,:] = np.roll(self.stim_drive, int(StimVals[sp])-int(self.S_N_neuron/2))
        
        #------------------------------------------------ 
        # flatten so can pass without looping over pools
        #------------------------------------------------                 
        return stims.flatten()
    
    #------------------------------------------------        
    #------------------------------------------------
    # Start a section of stuff to compute weights here....
    #------------------------------------------------
    #------------------------------------------------

    def get_recurrent_sensory_weights(self):
        """        
        Returns
        -------
        w : weight for the sensory neurons based on offset 'a'. If multiple pools
        of sensory neurons then this will return a single large matrix with 
        identical (but separate) weights for each pool to make it easier to 
        set up the synapses in the simulation. Note that matrix is flattened 
        per brian2 specifications

        """
        
        #------------------------------------------------         
        # from Burak and Fiete 2012 PNAS
        # numpy-fy to do all weights at once
        #------------------------------------------------ 
        a = np.linspace(0, (pi * 2)-((pi * 2) / self.S_N_neuron), self.S_N_neuron)

        #------------------------------------------------         
        # use meshgrid so can compute all weights at once
        #------------------------------------------------ 
        x,y = np.meshgrid(a,a)

        #------------------------------------------------ 
        # make all the weights here...
        #------------------------------------------------ 
        w_tmp = self.S_rec_w_baseline + self.S_rec_w_amp_exc * np.exp(self.S_rec_w_kappa_exc * (np.cos(x-y)-1)) - self.S_rec_w_amp_inh * np.exp(self.S_rec_w_kappa_inh * (np.cos(x-y)-1)) 
       
        #------------------------------------------------ 
        # no self excitation - so set main diag to 0
        #------------------------------------------------ 
        if self.S_SelfExcitation == False:
            d = np.diag_indices(self.S_N_neuron)
            w_tmp[d] = 0
            
        #------------------------------------------------ 
        # now assign w for each pool of sensory neurons to make one giant weight
        # matrix that has an identical set of weights for each pool
        # note: imshow this matrix to get a better idea of how it is put together.
        # When it is flattened the connection weights will be 0 between pools
        #------------------------------------------------ 
        w = np.zeros((self.S_N_neuron * self.S_N_pools, self.S_N_neuron * self.S_N_pools))
        for i in np.arange(self.S_N_pools):
            w[i * self.S_N_neuron : (i+1) * self.S_N_neuron, i * self.S_N_neuron : (i+1) * self.S_N_neuron] = w_tmp

        #------------------------------------------------ 
        # Now connect corresponding neurons across pools with reciprocal 
        # excitatory connections (if desired).
        # w is a self.S_N_neuron * self.S_N_pools by self.S_N_neuron * self.S_N_pools
        # matrix, so just need to fill the self.S_N_neuron * self.S_N_neuron sub
        # matrices with diag matrices to make these connections. 
        #------------------------------------------------ 
        if self.S_connect_pools:

            # figure out which neurons to connect between layers
            # based on specified proportion
            n_to_connect = np.random.choice([0, 1], size=self.S_N_neuron, p=[1-self.S_pools_to_S_pools_proportion, self.S_pools_to_S_pools_proportion])
            
            # then make diag matrix of weights to connect just corresponding 
            # neurons
            across_pool_w = np.diag(n_to_connect * self.S_pools_to_S_pools_w_amp)
            
            # first fill blocks in the upper diag portion of big matrix w
            # note todo: gotta be a better way to do upper and lower in one set 
            # of loops but going with this for now as only happens once...
            for i in np.arange(self.S_N_pools - 1):  
                for j in np.arange(1, self.S_N_pools):
                    w[i * self.S_N_neuron : (i+1) * self.S_N_neuron, j * self.S_N_neuron : (j+1) * self.S_N_neuron] = across_pool_w

            # then fill blocks in the lower diag portion of big matrix w
            for i in np.arange(1, self.S_N_pools):  
                for j in np.arange(self.S_N_pools-1):
                    w[i * self.S_N_neuron : (i+1) * self.S_N_neuron, j * self.S_N_neuron : (j+1) * self.S_N_neuron] = across_pool_w
                    

        ## have a look
        # plt.imshow(w)
        # plt.colorbar()
        # plt.show()
        
        #------------------------------------------------ 
        # flatten per brian2 convention and return
        #------------------------------------------------ 
        return w.flatten()
        
    def get_recurrent_rand_layer_weights(self):
        """        
        Returns
        -------
        w : weight for the rand layer neurons...inverted VM function
        so that nearby neurons inhibit each other the most and farther 
        away neurons don't interact...
    
        """
        
        #------------------------------------------------         
        # define a space to eval cos
        #------------------------------------------------ 
        a = np.linspace(0, (pi * 2)-((pi * 2) / self.R_N_neuron), self.R_N_neuron)
    
        #------------------------------------------------         
        # use meshgrid so can compute all weights at once
        #------------------------------------------------ 
        x,y = np.meshgrid(a,a)
    
        #------------------------------------------------ 
        # make all the weights here...
        #------------------------------------------------ 
        w = self.R_rec_w_baseline + self.R_rec_w_amp_exc * np.exp(self.R_rec_w_kappa_exc * (np.cos(x-y)-1)) - self.R_rec_w_amp_inh * np.exp(self.R_rec_w_kappa_inh * (np.cos(x-y)-1)) 
        
        # plt.imshow(w)
        # plt.show()
        
        #------------------------------------------------ 
        # no self excitation - so set main diag to 0
        #------------------------------------------------ 
        if self.R_SelfExcitation == False:
            d = np.diag_indices(self.R_N_neuron)
            w[d] = 0
        
        return w.flatten()
    
    def mask_w_sym_matrix(self, mask, norm_axis):
        """
        Mask the connection matrices to determine (1) which excitatory connections
        will exist, and if they exist, then give them a weight
        The same matrix will be used to mask excitatory connections for 
        feedforward and feedback directions (hense "symetric")
        
        Parameters
        ----------
        mask : np.array (size: num_sensory * pool x num_random)
            mask of excitatory connections
            
        norm_axis : boolean
            norm over sum of sensory neurons or sum of random neurons

        Returns
        -------
        m : np.array (size: num_sensory * pool x num_random)
        """
        
        #------------------------------------------------        
        # init mask as nans (will add baseline later)
        #------------------------------------------------     
        m = np.zeros(mask.shape)

        #------------------------------------------------        
        # Norm by sum over sensory neurons (True)
        # Norm by sum over random neurons (False)
        # Note: sum of a boolean array (mask) will yield the *count*
        # of the neurons with excitatory connections 
        # make all other entries equal to the baseline
        #------------------------------------------------ 
        if norm_axis==True:
            # do this clunky ind thing to avoid div by 0
            ind = np.sum(mask, axis=0)>0
            m[:,ind] = (self.weight_factor * self.S_to_R_target_w) / np.repeat(np.expand_dims(np.sum(mask[:,ind], axis=0), axis=0), mask.shape[0], axis=0)
            m[mask==False] = self.S_to_R_baseline

        else:
            ind = np.sum(mask, axis=1)>0
            m[ind,:] = (self.weight_factor * self.R_to_S_target_w) / np.repeat(np.expand_dims(np.sum(mask[ind,:], axis=1), axis=1), mask.shape[1], axis=1)
            m[mask==False] = self.R_to_S_baseline   
        
        return m
    
    
    def get_between_layer_weights(self):
        """
        Compute and balance feedforward and feedback weights between 
        sensory and random layers 
        
        Returns
        ------
        S_to_R_con_mat : np.array (S_N_neuron * S_N_pools matrix x R_N_neuron matrix)
            
        R_to_S_con_mat : np.array (R_N_neuron x S_N_neuron * S_N_pools matrix)
            
        """
        #------------------------------------------------                
        # initialize a matrix with rand(0,1)...
        # this controls the proportion of excitatory connections
        # between the sensory pools and the random layer
        # after just keeping entries < gamma (which is the param
        # that controls the proportion of excitatory connections)
        # 
        # Will use a circular normal to determine probability 
        # of connections between neurons in a given sensory pool
        # and neurons in the corresponding subset of the random layer
        # e.g. neurons in the first sensory pool will be 
        # probabilistically connected to the first R_N_neuron/S_N_pools
        # neurons in the rand layer, etc. (approximately...spread of 
        # circular normal could allow overlapping inputs to random neurons
        # from adjacent sensory pools)
        #------------------------------------------------        
        
        # init - total # of sensory neurons x total number of rand neurons
        sym_mat = np.random.rand(self.S_N_neuron * self.S_N_pools, self.R_N_neuron)    
        
        sym_mat = sym_mat < self.exc_con_prob
        
        # # axis to eval a circular normal 
        # x = np.linspace(0, (pi * 2)-((pi * 2) / self.R_N_neuron), self.R_N_neuron)
            
        # # number of neurons in rand layer that receive connections from neurons
        # # in each sensory pool...this is the step size for centering the circular
        # # normal top down spatial fields 
        # # check to make sure div evenly
        # if np.remainder(self.R_N_neuron, self.S_N_pools):
        #     print('Number of rand layer neurons must be evenly divisible by number of sensory pools')
        # else:
        #     Center_SF_Shift = int(self.R_N_neuron / self.S_N_pools)
        
        # # offset for position of first spatial field...
        # offset = int(Center_SF_Shift / 2)
        
        # # half-wave rectified and raised cos basis function
        # td_func = np.where(np.cos(x)<0,0,np.cos(x)) ** self.SF_exp
        
        # # normalize so that max is equal to exc_con_prob 
        # # or the probability of an excitatory connection
        # td_func = (td_func / np.max(td_func)) * self.exc_con_prob
        
        # # sum_spa_field = np.zeros(x.shape)
        # for i in np.arange(self.S_N_pools):
        #     tmp_basis = np.roll(td_func, (Center_SF_Shift * i) + offset)
        #     tmp_basis = np.repeat(np.expand_dims(tmp_basis,1).T, self.S_N_neuron, axis=0)
        #     # sum_spa_field += tmp_basis
        #     sym_mat[i * self.S_N_neuron : (i * self.S_N_neuron) + self.S_N_neuron, :] = sym_mat[i * self.S_N_neuron : (i * self.S_N_neuron) + self.S_N_neuron, :] < tmp_basis

        #------------------------------------------------        
        # sensory to random connection matrix
        # masked with sym_mat
        # this function will also set the baseline (if != 0)
        # output is a S_N_neuron * S_N_pools matrix x R_N_neuron matrix
        #------------------------------------------------        
        S_to_R_con_mat = self.mask_w_sym_matrix(sym_mat, True)
 
        #------------------------------------------------        
        # random to sensory connection matrix
        # masked with sym_mat
        # this function will also set the baseline (if != 0)
        # note: the output here is transposed to yield a 
        # R_N_neuron x S_N_neuron * S_N_pools matrix
        #------------------------------------------------      
        R_to_S_con_mat = self.mask_w_sym_matrix(sym_mat, False).T
            
        #------------------------------------------------ 
        # Balance feed forward connections from sensory to random layers
        # total excitation from sensory neurons to each neuron in random network...
        # then add ei factor, normalized by number of neurons in sensory layer, 
        # to sensory-to-random weights (feed forward weights)
        #------------------------------------------------     
        S_to_R_excitation = np.repeat(np.expand_dims(np.sum(S_to_R_con_mat, axis=0), axis=0), S_to_R_con_mat.shape[0], axis=0)
        S_to_R_con_mat += (self.S_to_R_ei_balance * S_to_R_excitation) / float(self.S_N_neuron * self.S_N_pools)
        
        #------------------------------------------------ 
        # Balance feedback connections from random to sensory layers
        # total excitation from random neurons to each neuron in sensory network...
        # then add ei factor, normalized by number of neurons in random layer, 
        # to random-to-sensory weights (feedback weights)
        # note: R_to_S_con_mat is a R_N_neuron x S_N_neuron * S_N_pools matrix
        # so we operate on the 0th axis just like with the S_to_R_con_mat matrix above 
        # in terms of np.repeat, np.sum, etc
        #------------------------------------------------     
        R_to_S_excitation = np.repeat(np.expand_dims(np.sum(R_to_S_con_mat, axis=0), axis=0), R_to_S_con_mat.shape[0], axis=0)
        R_to_S_con_mat += (self.R_to_S_ei_balance * R_to_S_excitation) / float(self.R_N_neuron)
        
        #------------------------------------------------ 
        # flatten per brian2 convention and return
        #------------------------------------------------ 
        return S_to_R_con_mat.flatten(), R_to_S_con_mat.flatten()
    
    #------------------------------------------------    
    #------------------------------------------------
    # Start a section of stuff to analyze data here
    #------------------------------------------------
    #------------------------------------------------

    def get_fr_vector(self, fr_mat, StimVals):
        """
        
        Parameters
        ----------
        fr_mat : brian2 units 
            vector of firing rates, in Hz, from a state monitor object
            (note that this function assumes Hz as the unit...)
            
        Returns
        -------
        fr_angle : np array
            angle of pop firing rate in each pool (in 'neuron' space)
            
        fr_abs : np array
            abs (magnitude) of pop firing rate in each pool 

        """
        
        # preferred features for each neuron (mu)
        n_mu = np.linspace(0, (pi * 2) - ((pi * 2) / self.S_N_neuron), self.S_N_neuron)
    
        # exp of neuron mu
        exp_n_mu = np.exp(1j * n_mu, dtype=complex)
    
        # abs (magnitude) and angle of firing rates 
        # at end of delay period
        fr_angle = np.zeros(self.S_N_pools)    
        fr_abs = np.zeros(self.S_N_pools)
        fr_mem_abs = np.zeros(self.S_N_pools)
  
        # loop over sensory pools...
        for sp in np.arange(self.S_N_pools) :
            
            # multiply the rate by the tuning defined around circle, undo the units by dividing (by Hz in this case)
            cmplx = np.mean(np.multiply(fr_mat[sp * self.S_N_neuron : (sp + 1) * self.S_N_neuron], exp_n_mu, dtype = complex)) / Hz
            
            # compute angle in 'neuron' space 
            fr_angle[sp] = np.angle(cmplx) * self.S_N_neuron / (2*pi)
            
            # compute the magnitude
            fr_abs[sp] = np.abs(cmplx)
            
            # then compute the magnitude of energy at remembered stim
            if ~np.isnan(StimVals[sp]):
                # multiply the rate by the tuning defined around circle, undo the units by dividing (by Hz in this case)
                cos_theta = np.cos(self.circ_diff(np.angle(cmplx), n_mu[int(StimVals[sp])]))
                fr_mem_abs[sp] = fr_abs[sp] * cos_theta  
    
        # wrap negative values
        fr_angle[fr_angle < 0] += self.S_N_neuron
    
        return fr_angle, fr_abs, fr_mem_abs
    
    
    def circ_diff(self, pred, actual):
        
        """
        Parameters
        ----------
        pred : np.array 
            Array of predicted (decoded) stimulus angles in radians.
        actual : np.array
            Array of actual stimulus angles in radians.

        Returns
        -------
        d : np.array
            Circular difference between pred and actual.

        Statistical analysis of circular data, N. I. Fisher
        Topics in circular statistics, S. R. Jammalamadaka et al. 
        Biostatistical Analysis, J. H. Zar
        
        See: Berens, https://github.com/circstat/circstat-matlab

        """
        
        d = np.angle(np.exp(1j * pred) / np.exp(1j * actual))
        
        return d    
    
    def circ_mean(self, x, ax, return_pos):
        
        """
        Parameters
        ----------
        x : np.array 
            Input array of stimulus angles in radians.
        ax : int
            Which dim of input array to operate on? 
            ax follows normal axis conventions for numpy

        Returns
        -------
        m : np.array
            Circular mean.

        Statistical analysis of circular data, N. I. Fisher
        Topics in circular statistics, S. R. Jammalamadaka et al. 
        Biostatistical Analysis, J. H. Zar
        
        See: Berens, https://github.com/circstat/circstat-matlab

        Note: Unlike the Berens implementation, this will not return
        negative values (instead it will return postive values near 2*pi).
        The behavior of this function should thus match scipy.stats.circmean

        """
        m = np.angle(np.nansum(np.exp(1j * x), axis = ax))
        

        if return_pos:
            # don't return negative angles...convert them
            # to always wrap counterclockwise around circle
            m = np.where(m < 0, (2 * pi) + m, m)
            
        return m
    
    #------------------------------------------------    
    #------------------------------------------------
    # A few functions for plotting connections and spikes
    #------------------------------------------------
    #------------------------------------------------
    def visualise_connectivity(self,S):
        """
        Helper function to visualize connectivity...
        adapted from brian2 docs but linewidth == abs(S.w)
        so you can also visualize the weights...
        Only use this with small networks - super slow!
        
        Parameters
        ----------
        S : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        Ns = len(S.source)
        Nt = len(S.target)
        # plt.figure(figsize=(10, 4))
        # plt.subplot(121)
        plt.plot(zeros(Ns), arange(Ns), 'ok', ms=10)
        plt.plot(ones(Nt), arange(Nt), 'ok', ms=10)
        for i, j in zip(S.i, S.j):
            plt.plot([0, 1], [i, j], '-k', linewidth=np.abs(S.w[i,j]))

        plt.xticks([0, 1], ['Source', 'Target'])
        plt.ylabel('Neuron index')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-1, max(Ns, Nt))
        plt.show()
    
    
    def plt_raster(self, S, y_label):
        """
        Raster plot of spikes over time
        todo: **kwargs to control colors, alpha, etc...
        
        Parameters
        ----------
        S : brian2 SpikeMonitor object
            
        y_label : string for ylabel, must be 'Sensory' or 'Random' 
            
        Returns
        -------
        None.

        """
        
        #------------------------------------------------        
        # decide if plotting sensory or random layer neurons
        #------------------------------------------------   
        if y_label=='Sensory':
            y_lim = self.S_N_neuron*self.S_N_pools
            plt_sensory = True
        else:
            y_lim = self.R_N_neuron
            plt_sensory = False         
        
        #------------------------------------------------        
        # do the plotting...
        #------------------------------------------------           
        fig, ax = plt.subplots()
        ax.plot(S.t/ms, S.i, '.k', ms=.5, alpha=.75)
        ax.set_ylim([0, y_lim])
        ax.set_ylabel(y_label + ' Neurons')
        ax.set_xlabel('Time (ms)')
        
        #------------------------------------------------        
        # plot some horizontal lines marking edges of each pool, only if plotting
        # sensory neurons
        #------------------------------------------------        
        if plt_sensory:
            for sp in np.arange(self.S_N_pools):
                ax.hlines(sp*self.S_N_neuron, 0, np.max(S.t/ms), 'k', linewidth = 1)
                
        #------------------------------------------------        
        # patch showing stim period and WM delay in different shades
        #------------------------------------------------        
        rect_stim = patches.Rectangle((0,0), self.StimExposeTime/ms, y_lim, linewidth=0, facecolor='g', alpha = .2)        
        rect_wm = patches.Rectangle((self.StimExposeTime/ms,0), (self.AttnTime/ms), y_lim, linewidth=0, facecolor='r', alpha = .2)        
        ax.add_patch(rect_stim)
        ax.add_patch(rect_wm)

            
        if plt_sensory:   
            plt.savefig('SensoryRaster-' + 'SetSize-' + str(self.SetSize) + '.jpg', dpi=600)
        else:
            plt.savefig('RandomRaster-' + 'SetSize-' + str(self.SetSize) + '.jpg', dpi=600)

        #------------------------------------------------        
        # show plot
        #------------------------------------------------                
        plt.show()

    
    #------------------------------------------------        
    # run a simulation of the two layer model
    # 1st layer is set of N sensory rings and 
    # 2nd layer is randomly and fully connected to all 
    # sensory rings 
    #------------------------------------------------ 
    
    def run_sim_rand(self):
        
        """
        Run N_trials simulations...
        """

        #------------------------------------------------        
        # turn on garbage collection...
        #------------------------------------------------
        gcPython.enable()

        #------------------------------------------------        
        # unpack these from self so that brian2 can find them...
        #------------------------------------------------
        S_tau = self.S_tau
        S_bias = self.S_bias
        S_InitSynRange = self.S_InitSynRange
        S_tf_slope = self.S_tf_slope
        R_tau = self.R_tau
        R_bias = self.R_bias
        R_InitSynRange = self.R_InitSynRange
        R_tf_slope = self.R_tf_slope
        
        #------------------------------------------------        
        # set up the neuron pools for the sensory neurons...
        #------------------------------------------------
        S_pools = NeuronGroup(self.S_N_neuron * self.S_N_pools, self.S_eqs, threshold='rand()<S_rate*dt', method = 'exact', dt = 0.1 * msecond)

        #------------------------------------------------
        # init each neuron with a random value between 0 and S_InitSynRange
        #------------------------------------------------
        S_pools.S_act = 'S_InitSynRange*rand()'   
        
        #------------------------------------------------        
        # Set up the recurent connections (synapses) for the sensory pools 
        # with synapse weights w and then connect all neurons to all 
        # other neurons
        #------------------------------------------------
        S_rec_con = Synapses(S_pools, S_pools, model='w : 1', on_pre='S_act+=w')  
        S_rec_con.connect()  

        #------------------------------------------------        
        # assign weights to sensory neurons
        #------------------------------------------------        
        S_rec_con.w = self.get_recurrent_sensory_weights()
                
        #------------------------------------------------        
        # for viewing connectivity weights...only do for small populations as 
        # a sanity check (takes forever to plt big networks)
        #------------------------------------------------                
        #self.visualise_connectivity(S_rec_con)
        
        #------------------------------------------------
        # set up a state monitor for sensory pool
        #------------------------------------------------
        S_state = StateMonitor(S_pools, 'S_rate', record=True, dt=self.TimeStepForRecording)

        #------------------------------------------------
        #------------------------------------------------
        # Now init the random layer...
        #------------------------------------------------
        #------------------------------------------------
        
        #------------------------------------------------        
        # set up the neuron group
        #------------------------------------------------
        R_pool = NeuronGroup(self.R_N_neuron, self.R_eqs, threshold='rand()<R_rate*dt', method = 'exact', dt = 0.1 * msecond)

        #------------------------------------------------
        # init each neuron with a random value between 0 and R_InitSynRange
        #------------------------------------------------
        R_pool.R_act = 'R_InitSynRange*rand()'   
        
        #------------------------------------------------        
        # Set up the recurent connections (synapses) for the rand layer
        # if we have specified that we want to have connections in this layer
        #------------------------------------------------
        if self.R_rec_w_amp_exc or self.R_rec_w_amp_inh:
            R_rec_con = Synapses(R_pool, R_pool, model='w : 1', on_pre='R_act+=w')  
            R_rec_con.connect()  
            
            #------------------------------------------------        
            # assign weights for random layer recurrent connections
            #------------------------------------------------        
            R_rec_con.w = self.get_recurrent_rand_layer_weights()        
        
        #------------------------------------------------
        # Connect neurons in the sensory pools with neurons in random pool 
        #------------------------------------------------    
        S_to_R_con = Synapses(S_pools, R_pool, model = 'w : 1', on_pre = 'R_act += w')
        S_to_R_con.connect()  
        
        #------------------------------------------------
        # Connect all neurons in the random pool with neurons in sensory pool 
        #------------------------------------------------        
        R_to_S_con = Synapses(R_pool, S_pools, model = 'w : 1', on_pre = 'S_act += w')
        R_to_S_con.connect()
        
        #------------------------------------------------
        # Get feedforward (sensory to random) and 
        # feedback (random to sensory) weights
        # and assign to synapses.
        #------------------------------------------------            
        S_to_R_con.w, R_to_S_con.w = self.get_between_layer_weights()
        
        #------------------------------------------------
        # set up a state monitor for random pool
        #------------------------------------------------
        R_state = StateMonitor(R_pool, 'R_rate', record=True, dt = self.TimeStepForRecording)

        #------------------------------------------------        
        # set up spike monitors for sensory and random pools
        #------------------------------------------------
        S_spike = SpikeMonitor(S_pools)
        R_spike = SpikeMonitor(R_pool)        
        
        #------------------------------------------------
        # store a snapshot of network after initialization
        #------------------------------------------------
        store('initialized')
                
        #------------------------------------------------
        # Loop over trials to run the sim!
        #------------------------------------------------
        print('running sim...')
        
        #------------------------------------------------        
        # init matrices to store decoded angle and magnitude (abs) on each trial 
        # in each pool and one more matrix to store the actual stim_values
        # and which pools were stimulated on the current trial
        #------------------------------------------------  
        nTimeSteps = int((self.StimExposeTime + self.AttnTime) / self.TimeStepForRecording)
        print('Time Steps: ', nTimeSteps)
        fr_angle = np.full((self.N_trials, self.S_N_pools, nTimeSteps), np.nan)
        fr_abs = np.full((self.N_trials, self.S_N_pools, nTimeSteps), np.nan)
        fr_mem_abs = np.full((self.N_trials, self.S_N_pools, nTimeSteps), np.nan)
        stim_vals = np.full((self.N_trials, self.S_N_pools), np.nan)
        stim_pools = np.full((self.N_trials, self.SetSize), np.nan)
        
        # two big matrices to store firing rates over time in the sensory and 
        # random networks
        S_fr = np.full((self.N_trials, self.S_N_pools * self.S_N_neuron, nTimeSteps), np.nan)
        R_fr = np.full((self.N_trials, self.R_N_neuron, nTimeSteps), np.nan)
        
        #------------------------------------------------
        # compute generic stimuls shape
        # do this once, and the roll this stim_drive function
        # based on StimVals on each trial
        #------------------------------------------------
        self.stim_drive = self.define_stims()       
        
        
        #------------------------------------------------
        # Start trial loop
        #------------------------------------------------
        for t in np.arange(self.N_trials):
            
            # ot = time.time()

            # print('Running trial:', t, 'Stim Strength:',self.StimStrength,'Rand:', self.R_stim, 'Number To Stim:', self.R_NumNeuronsToStim)
            print('Running trial:', t,'Number To Stim:', self.R_NumNeuronsToStim)
            #------------------------------------------------
            # restore the snapshot of the network on each trial
            #------------------------------------------------
            restore('initialized')
            
            #------------------------------------------------
            # Generate a list of stim vals for this trial
            # at whatever SetSize was specified - then store
            # them for later comparison to decoded values
            # this will randomly select SetSize pools to have a
            # stimulus, and will randomly select the stim
            #------------------------------------------------            
            stim_vals[t,:], stim_pools[t,:] = self.get_stim_vals()
            
            #------------------------------------------------
            # make a vector of stims for each pool based on 
            # StimVals for this trial (and which pools get a stim)
            #------------------------------------------------
            stims = self.roll_stims(stim_vals[t,:])
            # plt.plot(stims)

            #------------------------------------------------
            # generate baseline input levels for each pool
            # do on each trial because baseline input is generated
            # using np.random.rand (unless BaselineScale==0, 
            # in which case base_in will be zeros)
            #------------------------------------------------
            base_in = self.baseline_input()

            #------------------------------------------------
            # Then apply the stimuli to each pool
            #------------------------------------------------       
            S_pools.Stim = stims + base_in

            #------------------------------------------------
            # show the memory stims and run...
            #------------------------------------------------
            run(self.StimExposeTime)
            
            #------------------------------------------------
            # then reset the stim to sensory and rnd to baseline for the 'delay' period
            #------------------------------------------------
            # S_pools.Stim = base_in

            #------------------------------------------------
            # Then apply stim to the random layer
            # to the specified number of neurons based on their
            # activity level
            #------------------------------------------------    
            if self.R_StimProportion: # set this to 1 if uniform gain
                
                if self.rand_top_down == 1: # if a random subset of neurons will get top down stim
                    ind = np.arange(0, self.R_N_neuron)
                    np.random.shuffle(ind)
                    R_pool.Stim[ind[:self.R_NumNeuronsToStim]] = self.R_stim
                    # print(self.R_stim)
                    # print(ind[:self.R_NumNeuronsToStim])
                    # print(R_pool.Stim[ind[:self.R_NumNeuronsToStim]])
                    # print(R_state.R_rate[ind[:self.R_NumNeuronsToStim],-1])

                    print('Rand TD Stim', 'SetSize:', self.SetSize)

                    
                elif self.rand_top_down == 0: # apply stim to top most feedforward activation neurons
                    sort_ind = np.argsort(R_state.R_rate[:,-1], axis=0) # current state of firing
                    # plt.plot(R_state.R_rate[sort_ind[self.R_N_neuron - self.R_NumNeuronsToStim : -1],-1])
                    # print(R_state.R_rate.shape)
                    # print(R_state.R_rate[sort_ind[self.R_N_neuron - self.R_NumNeuronsToStim : -1],-1])
                    stim_ind = np.zeros(self.R_N_neuron)
                    stim_ind[sort_ind[self.R_N_neuron - self.R_NumNeuronsToStim : -1]] = self.R_stim
                    R_pool.Stim = stim_ind
                    # print('Normal TD stim')
                    # print(self.R_stim)
                    # print(sort_ind[:self.R_NumNeuronsToStim])
                    # print(R_state.R_rate[sort_ind[self.R_N_neuron - self.R_NumNeuronsToStim : -1],:])
                    # print('No Rand TD Stim', 'SetSize:', self.SetSize)
                    
                else:
                    print('specify real top down modulation proportion')
                    
            # #------------------------------------------------
            # # wait the WM delay period
            # #------------------------------------------------
            run(self.AttnTime)

            #------------------------------------------------
            # if stim the R pool then reset to zero
            #------------------------------------------------
            if self.R_stim: 
                R_pool.Stim = np.zeros(self.R_N_neuron)
                S_pools.Stim = base_in
                
            # trial timer
            # print(time.time() - ot)

            #------------------------------------------------
            # get predicted ang,abs of vector over each time window (tw) in the trial for each pool...
            # use the firing rates computed and stored in the StateMonitor(S_state)
            #------------------------------------------------
            for tw in np.arange(nTimeSteps):
                fr_angle[t,:,tw], fr_abs[t,:,tw], fr_mem_abs[t,:,tw] = self.get_fr_vector(S_state.S_rate[:,tw], stim_vals[t,:])
            temp = S_state.S_rate[:,-1]
            #------------------------------------------------
            # Save the firing rates in the sensory and random layers
            #------------------------------------------------            
            S_fr[t,:,:] = S_state.S_rate
            R_fr[t,:,:] = R_state.R_rate
            
        #------------------------------------------------
        # simple raster plot spikes in each pool over time
        # First the sensory pools and then the random layer
        # This is just a demo of plotting...only does the last 
        # trial
        #------------------------------------------------
        
        # self.plt_raster(S_spike, 'Sensory') # only run if trial is one
        # self.plt_raster(R_spike, 'Random')
        
        #------------------------------------------------        
        # finish up garbage collection...only if not on mac
        #------------------------------------------------
        #if sys.platform != 'darwin':
        gcPython.collect()

        return fr_angle, fr_abs, fr_mem_abs, stim_vals, stim_pools, S_fr, R_fr, temp

