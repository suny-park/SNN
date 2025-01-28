#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:10:27 2025

Helper functions for analyze_feature.py

@author: supark
"""
from sklearn.multioutput import RegressorChain
from sklearn.linear_model import BayesianRidge
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

#------------------------------------------------        
#------------------------------------------------
# Compute circular regression with ridge...
# empirical bayes estimataion of alpha (lambda)
# uses chaining to achieve multi-output results
#------------------------------------------------
#------------------------------------------------ 
def compute_reg_weights( X, y ):
    
    '''
    
    Circular regression with empirical bayes ridge
    
    Parameters
    ----------
    X : Trial x voxel matrix
    y : list of angles in degrees

    Returns
    -------
    w : weights on each voxel after BayesianRidge regression with 
        multi-ouput supported by chaining
        
    '''
    
    new_y = y.copy() 
    
    #------------------------------------------------
    # ang in degrees into sin/cos
    #------------------------------------------------
    trn_y = np.vstack( ( np.sin( np.pi *  new_y / 180 ),  np.cos( np.pi * new_y / 180 ) ) ).T

    #------------------------------------------------
    # make the model - can add alternatives here as well
    # for non-circular data (e.g. multinomial logistic, etc)
    #------------------------------------------------
    model = BayesianRidge()

    # define chain for multi-output 
    chain = RegressorChain( model )
    
    # grab the data over the desired time window, then mean over
    # that window
    # trnX = np.mean( X[ :, self.avg_window[ self.train_type ][ 0 ]:self.avg_window[ self.train_type ][ 1 ], : ], axis = 1 )
    trnX = X
    
    # train model
    my_model = chain.fit( trnX, trn_y )
    
    # return model(s)
    return my_model 


#------------------------------------------------ 
#------------------------------------------------
# decode using regression...
#------------------------------------------------
#------------------------------------------------ 
def decode_reg( my_model, tst_data ):

    '''
    
    Predict (Decode) using regression
    
    Parameters
    ----------
    my_models : trained regression model(s)
    X : Trial x voxel matrix

    Returns
    -------
    y_hat : prediction!
    
    '''

    # mean over avg time window...
    # tmp_tst_data = np.mean( tst_data[ :, self.avg_window[ self.test_type ][ 0 ]:self.avg_window[ self.test_type ][ 1 ], : ], axis = 1 )
    tmp_tst_data = tst_data

    # get predictions
    tmp_hat = my_model.predict( tmp_tst_data )
    
    # convert back to angle in rads
    y_hat = np.arctan2( tmp_hat[:,0], tmp_hat[:,1] ) 

    # wrap any neg values back to positive to make plotting nice...
    y_hat[ y_hat<0 ] = ( 2*np.pi ) + y_hat[ y_hat<0 ] 

    return y_hat

def wrapRad(d): # wraps to +/- pi
    d[np.abs(d)>pi]-= 2*pi*np.sign(d[np.abs(d)>pi])
    return d

def circ_MAE(y,y_hat):
    MAE = np.mean(abs(wrapRad(y_hat-y)))
    return MAE

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