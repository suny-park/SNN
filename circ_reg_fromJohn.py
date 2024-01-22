#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:08:29 2023

@author: johnserences
"""

from sklearn.multioutput import RegressorChain
from sklearn.linear_model import BayesianRidge
import numpy as np

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
