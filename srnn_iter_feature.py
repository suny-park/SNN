from brian2 import *
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from scipy.special import i0, i1 
import time

from random import choices # needed for sampling from pdf

from scipy.io import savemat


import gc as gcPython

import cython
prefs.codegen.target = 'auto'
prefs.codegen.cpp.compiler = 'unix'
prefs.core.default_float_dtype = float32

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
        self.StimExposeTime = params.get('StimExposeTime', .3 * second)
        self.BaselineScale = params.get('BaselineScale', 0)
        self.stim_strengths = params.get('stim_strengths', [0,2,4,6,8])
        self.kappa = params.get('kappa', 14)
        self.StimToZero = params.get('StimToZero', 3)
        
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
        self.N_trials_sensory = params.get('N_trials_sensory', 1)
        self.N_trials_attention = params.get('N_trials_attention', 1)
        
        
        #------------------------------------------------ 
        # random seed - this is important! will determine
        # all subsequent output!
        #------------------------------------------------ 
        self.rndSeed = params.get('rndSeed', 0)
        np.random.seed(self.rndSeed)
        seed(self.rndSeed)
        
        
        self.doPlot = params.get('doPlot',0)

        self.N_stim_loc = params.get('N_stim_loc',16)
        self.N_stim_main = params.get('N_stim_main',2)

        self.r_stim_amps = params.get('r_stim_amps',[5])
        self.r_stim_ratios = params.get('r_stim_ratios',[0.2])

        self.shiftStep = params.get('shiftStep', 32)
        self.shiftReps= params.get('shiftReps', 1)

        self.rand_kappa = params.get('rand_kappa',0)

        self.saveSpikeDat = params.get('saveSpikeDat',0)
         
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
        # print('Baseline: ',self.BaselineScale, ' base_in: ', base_in[0,0], base_in[1,0])


        #------------------------------------------------ 
        # flatten so can pass without looping over pools
        #------------------------------------------------ 
        return base_in.flatten()
    
    
    def define_stims(self,stim_strength):
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
        # stim = self.StimStrength * stim
        stim = stim_strength * stim

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
                # print(StimVals[sp])
                stims[sp,:] = np.roll(self.stim_drive, int(StimVals[sp])-int(self.S_N_neuron/2))
        
        #------------------------------------------------ 
        # flatten so can pass without looping over pools
        #------------------------------------------------                 
        return stims.flatten()
    
    def roll_stims_main(self,plotStim,attStimShift):
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
        # when we're presenting 2 stimuli to pool 0 and 1 stimulus to pool 1, and applying attention to one of pool 0 stimulus
        # loop over Pool 0 stimulus values and add stimulus input after rolling
        for ps in range(self.N_stim_main):
            # stims[0,:] = stims[0,:]+np.roll(self.stim_drive, int(self.S_N_neuron/2*ps+self.S_N_neuron/8*2 - self.S_N_neuron/2))
            # change how much to shift based on the rep value
            stims[0,:] = stims[0,:]+np.roll(self.stim_drive, int(self.S_N_neuron/2*ps+self.S_N_neuron/self.shiftStep*attStimShift) - int(self.S_N_neuron/2))
        if plotStim==1:
            plt.plot(stims[0,:])
            plt.show()
            
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
        # for a given random neuron, weights are scaled by the total number of excitatory connections
        if norm_axis==True:
            # do this clunky ind thing to avoid div by 0
            ind = np.sum(mask, axis=0)>0
            m[:,ind] = (self.weight_factor * self.S_to_R_target_w) / np.repeat(np.expand_dims(np.sum(mask[:,ind], axis=0), axis=0), mask.shape[0], axis=0)
            m[mask==False] = self.S_to_R_baseline
        # for a given sensory neuron, weights are scaled by the total number of excitatory connections
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
        #------------------------------------------------        
        
        # init - total # of sensory neurons x total number of rand neurons
        # used as a mask to make the actual weight matrix
        sym_mat = np.zeros((self.S_N_neuron * self.S_N_pools, self.R_N_neuron))
        
        ####################################################################
        # parametrically modulate randomness of the connections between sensory and random layer neurons
        # decide the number of neurons to be connected first for each pool and then draw from pdf
        # this is to match the number of connections to the previous version
        R_to_S_pool = np.random.rand(self.S_N_pools, self.R_N_neuron) # matrix that defines which pools will have connections to each R neurons
        R_to_S_pool = R_to_S_pool < self.exc_con_prob # probablity of connection - still using gamma
        
        x = np.linspace(0, (pi * 2)-((pi * 2) / self.S_N_neuron), self.S_N_neuron)
        
        # plot to make sure the shape of the probability distribution looks right
        if self.doPlot:
            circ_norm_func = np.exp(self.rand_kappa * np.cos(x-pi))/(2 * pi * i0(self.rand_kappa)) 
            circ_norm_func /= np.sum(circ_norm_func) # normalize to amp = 1
            plt.plot(circ_norm_func)
            plt.title('kappa: '+str(self.rand_kappa))
            plt.show()
        
        for rn in np.arange(self.R_N_neuron): # looping through every single random layer neurons
            for sp in np.arange(self.S_N_pools): # for each random neuron, let's loop through each sensory pool
                thispoolconn = np.random.rand(self.S_N_neuron, 1)
                thispoolconn = thispoolconn < self.exc_con_prob # only keep the ones that are < gamma
                n_thispoolconn = np.sum(thispoolconn) # number of connections for this pool
                if n_thispoolconn > 0: # if there are any connections for this pool
                    # if there was no connection for this neuron up to this point
                    if np.sum(sym_mat[:,rn]) == 0: 
                        mu_idx = int(np.random.randint(self.S_N_neuron) ) # randomly draw a sensory neuron within this pool to be connected
                        # set up a probability distribution centered on the first chosen preferred stimulus
                        circ_norm_func = np.exp(self.rand_kappa * np.cos(x-x[mu_idx]))/(2 * pi * i0(self.rand_kappa)) 
                        circ_norm_func /= np.sum(circ_norm_func) # normalize to amp = 1
                    # if there are already connections for this neuron, use the already set-up probability distribution function to determine 
                    # which sensory units in this pool to connect to
                    allconns = np.random.choice(x,size=n_thispoolconn,replace=False, p=circ_norm_func)
                    # loop through all connections and actually record in the matrix mask
                    for thisconn in allconns:
                        conn_idx = int(np.argwhere(x==thisconn))
                        sym_mat[sp*self.S_N_neuron+conn_idx,rn] = 1
                
        sym_mat = sym_mat==1
        
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
    
    
    def plt_raster_sen(self, S, y_label):
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
        ax.set_title('StimStr: 10')
        
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
        # rect_wm = patches.Rectangle((self.StimExposeTime/ms,0), (self.AttnTime/ms), y_lim, linewidth=0, facecolor='r', alpha = .2)        
        ax.add_patch(rect_stim)
        # ax.add_patch(rect_wm)

        #-------------------------------------------
        # save plot
        #-------------------------------------------
        plt.savefig('F_SensoryTask_'+y_label+'_StimStrength-10.png', dpi=600)
        # print((S.t/ms).shape)
        # print(S.i.shape)
        # from scipy.io import savemat
        # raster_dic = {'time':S.t/ms,'spikes':S.i}
        # savemat('raster_test.mat',raster_dic)

        #------------------------------------------------        
        # show plot
        #------------------------------------------------                
        plt.show()
        
    def plt_raster_att(self, S, y_label,attStim):
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
        ax.set_title('Att:'+str(attStim)+' R_stim:'+str(self.R_stim)+' StimStr:'+str(self.StimStrength))
        
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
        rect_stim = patches.Rectangle((0,0), self.AttnTime/ms, y_lim, linewidth=0, facecolor='r', alpha = .2)        
        ax.add_patch(rect_stim)
        
        #-------------------------------------------
        # save plot
        #-------------------------------------------
        plt.savefig('F_AttentionTask_'+y_label+'_Att-'+str(attStim)+'_R_stim-'+str(self.R_stim)+'_StimStrength-'+str(self.StimStrength)+'.png', dpi=600)

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
        #print('running sim...')
        
        #------------------------------------------------        
        # init matrices to store decoded angle and magnitude (abs) on each trial 
        # in each pool and one more matrix to store the actual stim_values
        # and which pools were stimulated on the current trial
        #------------------------------------------------  
        # nTimeSteps = int(self.StimExposeTime / self.TimeStepForRecording)
        nTimeSteps = int((self.AttnTime) / self.TimeStepForRecording)
        
        
        # 1) how similar are the top 10% random neurons selected each time after restoring initialized state
        # : depends on the % of stimulated random neurons, but 30% and higher
        # 2) out of curiosity, compare "restore('initialized')" with resetting to baseline "S_pools.Stim = base_in"
        # : pretty similar, but be careful as restore resets saved firing rates and won't give you the full raster plot of the whole trial
        
        #------------------------------------------------
        # Start trial loop -- run with target stim in target pool to define most active random neurons for that stimulus,
        # reset things back to baseline, stimulate target and second pool with top-down gain on proportion of most active random neurons
        #------------------------------------------------
        
        ### 1. Sensory Task ###
        # present single stimulus to each of the pools and record the response over N trials
        # try presenting the same stimulus across all pools
        # output: trial label of presented stimulus, response (trial X neuron)
        # need to loop over stimulus
        
        #------------------------------------------------
        # generate baseline input levels for each pool
        # do on each trial because baseline input is generated
        # using np.random.rand (unless BaselineScale==0, 
        # in which case base_in will be zeros)
        #------------------------------------------------
        base_in = self.baseline_input()
        
        #%% 
        
        print('Phase 1: Stimulating Sensory Pools')
        
        N_stim_loc = self.N_stim_loc # number of stimuli in the localizer task
        nTotalTrials = self.N_trials_sensory*N_stim_loc*self.S_N_pools
        S_fr_avg_loc = np.full((nTotalTrials, self.S_N_pools * self.S_N_neuron), np.nan)
        label_stim_loc = np.full((nTotalTrials), np.nan)
        label_pool_loc = np.full((nTotalTrials), np.nan)
        label_trial_loc = np.full((nTotalTrials), np.nan)
        
        tc = 0 # global trial counter for localizer task
        
        # present one stimulus to one pool at a time
        
        # define an empty numpy object to draw raster plots later
        S_SensoryTask = np.empty((N_stim_loc, self.S_N_pools), dtype=object)
        R_SensoryTask = np.empty((N_stim_loc, self.S_N_pools), dtype=object)
        
        for whichstim in np.arange(N_stim_loc):
            
            for stimPool in range(self.S_N_pools):
            
                # present "whichstim" to "stimPool"
                stim_vals = np.full(self.S_N_pools, np.nan)
                thisStim = int(self.S_N_neuron/N_stim_loc*whichstim)
                stim_vals[stimPool] = thisStim # should range 0-512
                
                # compute generic stimulus shape
                self.stim_drive = self.define_stims(10) # stim_strength manually set to 10
                
                # make a vector of stims for each pool
                stims = self.roll_stims(stim_vals) # S_N_pools x S_N_neuron matrix of stims
                
                # run N trials for the given stimulus
                for t in np.arange(self.N_trials_sensory):
                    
                    #------------------------------------------------
                    # restore the snapshot of the network on each trial
                    #------------------------------------------------
                    restore('initialized')
                    
                    base_in = self.baseline_input()
                    
                    #------------------------------------------------
                    # Then apply the stimuli to each pool
                    #------------------------------------------------       
                    S_pools.Stim = stims + base_in
        
                    #------------------------------------------------
                    # show the to-be attended stims and run...
                    #------------------------------------------------
                    run(self.StimExposeTime)
                    
                    # record the response across all pools 
                    # #------------------------------------------------
                    # # Save the firing rates in the sensory and random layers
                    # #------------------------------------------------            
                    S_fr_avg_loc[tc,:] = np.mean(S_state.S_rate[:,-5:],axis=1) # average FR for the last 5 timesteps (trial,neurons)
                    label_stim_loc[tc] = thisStim # presented stimulus label
                    label_pool_loc[tc] = stimPool
                    label_trial_loc[tc] = t
                    
                    tc+=1 # increase global trial counter
                    
                    if self.doPlot:
                        if t == 0: # only for the first trial
                            self.plt_raster_sen(S_spike,'Sensory')

                    if self.saveSpikeDat and (t == 0):
                        timepoints = np.array(S_spike.t/ms)
                        spikes = np.array(S_spike.i)
                        spike_arr = np.vstack((timepoints, spikes))
                        S_SensoryTask[whichstim,stimPool] = spike_arr
                        
                        timepoints = np.array(R_spike.t/ms)
                        spikes = np.array(R_spike.i)
                        spike_arr = np.vstack((timepoints, spikes))
                        R_SensoryTask[whichstim,stimPool] = spike_arr
        
        
        
        ### 2. Attention Task ###
        # present two stimuli to the first pool and apply top-down gain
        # record response from the last time windows 
        # output: trial label of attended stimulus, response (trial X neuron)
        # Step 1: Define random neurons that maximally respond to to-be-attended stimulus
        # Step 2: Present two stimuli and apply gain to one of them, record activity 
        print('')
        print('Phase 2: Applying Gain to Random Layer')
        
        nTotalTrial = self.N_trials_attention*self.N_stim_main*self.shiftReps*len(self.stim_strengths)
        
        S_fr_avg_main = np.full((nTotalTrial, self.S_N_pools * self.S_N_neuron,len(self.r_stim_amps),len(self.r_stim_ratios)), np.nan)
        
        # define an empty numpy object to draw raster plots later
        S_AttentionTask = np.empty((len(self.r_stim_amps), len(self.stim_strengths), self.N_stim_main), dtype=object)
        R_AttentionTask = np.empty((len(self.r_stim_amps), len(self.stim_strengths), self.N_stim_main), dtype=object)
        
        for r_cnt, R_stim in enumerate(self.r_stim_amps):
            self.R_stim = R_stim
            
            for r_prop, R_StimProportion in enumerate(self.r_stim_ratios):
                
                if R_stim == 0:
                    R_StimProportion = 0
                    
                tc = 0 # global trial counter for main task
                label_stim_main = np.full((nTotalTrial), np.nan) #these should be the same across r_stim loop
                label_trial_main = np.full((nTotalTrial), np.nan) #these should be the same across r_stim loop
                label_stim_strength_main = np.full((nTotalTrial), np.nan) # stimulus strength
                
                for ss_cnt, self.StimStrength in enumerate(self.stim_strengths):
                
                    # another loop here for shifting stimulus pairs
                    # for attStimShift in np.arange(self.shiftReps):
                    for attStimShift in [8]: # fixed at 8 for better visualization
                    
                        for attStim in range(self.N_stim_main):
                            
                            # loop over trials of [defining R neurons + applying gain]
                            for t in np.arange(self.N_trials_attention):
                                
                                stim_vals = np.full((self.S_N_pools), np.nan)
                                
                                # present to-be-attended stimulus in the first sub-network to select which neurons to apply top-down gain
                                attStimVal = int(self.S_N_neuron/self.N_stim_main*attStim)+int(self.S_N_neuron/self.shiftStep*attStimShift) 
                                stim_vals[0] = attStimVal
                                
                                # compute generic stimulus shape
                                self.stim_drive = self.define_stims(10) # stim_strength manually set to 10
                                
                                # roll to the right place
                                stims = self.roll_stims(stim_vals) # S_N_pools x S_N_neuron matrix of stims
                                
                                # reset network + add baseline
                                restore('initialized')
                                base_in = self.baseline_input()
                                
                                # present stimulus
                                S_pools.Stim = stims + base_in
                                run(self.StimExposeTime)
                                # sort random neurons by their firing rate
                                sort_ind = np.argsort(R_state.R_rate[:,-1], axis=0) # current state of firing
                                stim_ind = np.zeros(self.R_N_neuron)
                                R_NumNeuronsToStim = int(np.floor(R_StimProportion * self.R_N_neuron))
                                stim_ind[sort_ind[self.R_N_neuron - R_NumNeuronsToStim : -1]] = R_stim
                                
                                # reset network
                                restore('initialized')
                                S_pools.Stim = base_in
                                run(self.StimExposeTime)
                                restore('initialized')
                                
                                self.stim_drive = self.define_stims(self.StimStrength) # define again with actual stimulus strength
                                # present two stimuli at the same time
                                plotStim  = 0
                                stims = self.roll_stims_main(plotStim,attStimShift)
                                
                                # apply stimulus & top-down gain
                                S_pools.Stim = stims + base_in
                                R_pool.Stim = stim_ind
                                run(self.AttnTime)
                                
                                # record FR
                                S_fr_avg_main[tc,:,r_cnt,r_prop] = np.mean(S_state.S_rate[:,-5:],axis=1) # average FR for the last 5 timesteps (trial,neurons)
                                label_stim_main[tc] = attStimVal # attended stimulus label
                                label_trial_main[tc] = t
                                label_stim_strength_main[tc] = self.StimStrength
                                
                                if self.doPlot:
                                    if t == 0: # only for the first trial
                                        self.plt_raster_att(S_spike, 'Sensory',attStim)
                                        
                                    
                                if self.saveSpikeDat and (t == 0):
                                    timepoints = np.array(S_spike.t/ms)
                                    spikes = np.array(S_spike.i)
                                    spike_arr = np.vstack((timepoints, spikes))
                                    S_AttentionTask[r_cnt,ss_cnt,attStim] = spike_arr
                                    
                                    timepoints = np.array(R_spike.t/ms)
                                    spikes = np.array(R_spike.i)
                                    spike_arr = np.vstack((timepoints, spikes))
                                    R_AttentionTask[r_cnt,ss_cnt,attStim] = spike_arr
                                
                                
                                tc+=1 # increase global trial counter

        if self.saveSpikeDat:
            # save out spike info for drawing raster plots                        
            raster_dic = {'S_SensoryTask':S_SensoryTask,'R_SensoryTask':R_SensoryTask, \
                          'S_AttentionTask':S_AttentionTask,'R_AttentionTask':R_AttentionTask,\
                              'stim_strengths':self.stim_strengths,'r_stim_amps':self.r_stim_amps,\
                                  'keys':'Sensory[stim,pool],Attention[r_stim_amp,stim_strength,attstim]'}
            savemat('results/F_spikes_kappa-'+str(self.rand_kappa)+'.mat',raster_dic)
            print('Saved results/F_spikes_kappa-'+str(self.rand_kappa)+'.mat')
        
        #------------------------------------------------        
        # finish up garbage collection...only if not on mac
        #------------------------------------------------
        #if sys.platform != 'darwin':
        gcPython.collect()

        return S_fr_avg_loc, label_stim_loc, label_pool_loc, label_trial_loc, S_fr_avg_main, label_stim_main, label_trial_main, label_stim_strength_main

