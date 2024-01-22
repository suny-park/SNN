import numpy as np
from numpy import pi

def circ_corr_pval(x,yi,nComp,get_full=False):
    y = yi.copy()
    true = circ_corr(x, y)
    shuf = np.zeros(nComp)
    for i in range(nComp):
        np.random.shuffle(y)
        shuf[i] = circ_corr(x, y)
    if get_full:
        return true,shuf
    else:
        return true,np.mean(true<shuf)
    

def circ_corr(x,y):
    '''
    calculate correlation coefficient between two circular variables
    using fisher & lee circular correlation formula (code from Ed Vul)
    x, y are both in radians [0.2pi]
    '''
    xu = x.copy()
    yu = y.copy()
    if np.any(x>90): xu = xu.astype(float); xu/=90*np.pi
    if np.any(y>90): yu = yu.astype(float); yu/=90*np.pi
    n 	= len(x)
    A 	= np.sum( np.multiply( np.cos( xu ), np.cos( yu ) ) )
    B 	= np.sum( np.multiply( np.sin( xu ), np.sin( yu ) ) )
    C 	= np.sum( np.multiply( np.cos( xu ), np.sin( yu ) ) )
    D 	= np.sum( np.multiply( np.sin( xu ), np.cos( yu ) ) )
    
    E 	= np.sum( np.cos( 2*xu ) )
    F 	= np.sum( np.sin( 2*xu ) )
    G 	= np.sum( np.cos( 2*yu ) )
    H 	= np.sum( np.sin( 2*yu ) )
    corr_coef = 4*( A*B - C*D ) / np.sqrt( ( n**2 - E**2 - F**2 ) * ( n**2 - G**2 - H**2 ) )
    return corr_coef

def circ_corrW(x,y,ws):
    '''
    calculate correlation coefficient between two circular variables
    using fisher & lee circular correlation formula (code from Ed Vul)
    x, y are both in radians [0.2pi]
    '''
    xu = x.copy()
    yu = y.copy()
    w  = ws.copy() 
    w  = (w-np.min(w))/np.max(w)
    w += 1-np.mean(w)
    if np.any(x>90): xu = xu.astype(float); xu/=90*np.pi
    if np.any(y>90): yu = yu.astype(float); yu/=90*np.pi
    n 	= len(x)
    A 	= np.sum( w*np.multiply( np.cos( xu ), np.cos( yu ) ) )
    B 	= np.sum( w*np.multiply( np.sin( xu ), np.sin( yu ) ) )
    C 	= np.sum( w*np.multiply( np.cos( xu ), np.sin( yu ) ) )
    D 	= np.sum( w*np.multiply( np.sin( xu ), np.cos( yu ) ) )
    
    E 	= np.sum( np.cos( 2*xu ) )
    F 	= np.sum( np.sin( 2*xu ) )
    G 	= np.sum( np.cos( 2*yu ) )
    H 	= np.sum( np.sin( 2*yu ) )
    corr_coef = 4*( A*B - C*D ) / np.sqrt( ( n**2 - E**2 - F**2 ) * ( n**2 - G**2 - H**2 ) ) / np.sqrt(np.mean(w**2))
    return corr_coef

def wrapRad(d): # wraps to +/- pi
    d[np.abs(d)>pi]-= 2*pi*np.sign(d[np.abs(d)>pi])
    return d

def circ_MSE(y,y_hat):
    MSE = np.mean(wrapRad(y_hat-y)**2)
    return MSE

def circ_MAE(y,y_hat):
    MAE = np.mean(abs(wrapRad(y_hat-y)))
    return MAE