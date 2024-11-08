#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 19:45:22 2024

@author: supark
"""
from brian2 import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

def plt_raster_sen(S,S_N_neuron,S_N_pools,StimStrength,StimExposeTime):
    """
    Raster plot of sensory layer spikes over time for sensory task
    todo: **kwargs to control colors, alpha, etc...
    
    Parameters
    ----------
    S : brian2 SpikeMonitor object
                
    Returns
    -------
    None.

    """
    
    #------------------------------------------------        
    # plotting sensory layer neurons
    #------------------------------------------------   
    y_lim = S_N_neuron*S_N_pools
    y_label = 'Sensory'
    
    #------------------------------------------------        
    # do the plotting...
    #------------------------------------------------           
    fig, ax = plt.subplots()
    ax.plot(S.t/ms, S.i, '.k', ms=.5, alpha=.75)
    ax.set_ylim([0, y_lim])
    ax.set_ylabel(y_label + ' Neurons')
    ax.set_xlabel('Time (ms)')
    ax.set_title('StimStr:'+str(StimStrength))
    
    #------------------------------------------------        
    # plot some horizontal lines marking edges of each pool, only if plotting
    # sensory neurons
    #------------------------------------------------        
    # if plt_sensory:
    for sp in np.arange(S_N_pools):
        ax.hlines(sp*S_N_neuron, 0, np.max(S.t/ms), 'k', linewidth = 1)
            
    #------------------------------------------------        
    # patch showing stim period and WM delay in different shades
    #------------------------------------------------        
    rect_stim = patches.Rectangle((0,0), StimExposeTime/ms, y_lim, linewidth=0, facecolor='g', alpha = .2)        
    # rect_wm = patches.Rectangle((self.StimExposeTime/ms,0), (self.AttnTime/ms), y_lim, linewidth=0, facecolor='r', alpha = .2)        
    ax.add_patch(rect_stim)
    # ax.add_patch(rect_wm)

    #------------------------------------------------        
    # if ping, show it...
    #------------------------------------------------  
    # if self.PingStimScale > 0:
    #     rect_ping = patches.Rectangle(((self.StimExposeTime+self.PingStimOnset)/ms,0), (self.PingExposeTime/ms), y_lim, linewidth=0, facecolor='k', alpha = .2)        
    #     ax.add_patch(rect_ping)
        
    # if plt_sensory:   
    #     plt.savefig('SensoryRaster-' + 'SetSize-' + str(self.SetSize) + '.jpg', dpi=600)
    # else:
    #     plt.savefig('RandomRaster-' + 'SetSize-' + str(self.SetSize) + '.jpg', dpi=600)

    #------------------------------------------------        
    # show plot
    #------------------------------------------------                
    plt.show()
    
def plt_raster_att(S,attStim,S_N_neuron,S_N_pools,StimStrength,StimExposeTime):
    """
    Raster plot of sensory layer spikes over time for attention task
    todo: **kwargs to control colors, alpha, etc...
    
    Parameters
    ----------
    S : brian2 SpikeMonitor object
        
        
    Returns
    -------
    None.

    """
    
    #------------------------------------------------        
    # decide if plotting sensory or random layer neurons
    #------------------------------------------------   
    y_lim = S_N_neuron*S_N_pools
    y_label = 'Sensory'
    
    #------------------------------------------------        
    # do the plotting...
    #------------------------------------------------           
    fig, ax = plt.subplots()
    ax.plot(S.t/ms, S.i, '.k', ms=.5, alpha=.75)
    ax.set_ylim([0, y_lim])
    ax.set_ylabel(y_label + ' Neurons')
    ax.set_xlabel('Time (ms)')
    
    ax.set_title('Att:'+str(attStim)+' R_stim_S:'+str(self.R_stim_S)+' R_stim_F:'+str(self.R_stim_F)+' StimStr:'+str(self.StimStrength))
    
    #------------------------------------------------        
    # plot some horizontal lines marking edges of each pool, only if plotting
    # sensory neurons
    #------------------------------------------------        
    for sp in np.arange(self.S_N_pools):
        ax.hlines(sp*self.S_N_neuron, 0, np.max(S.t/ms), 'k', linewidth = 1)

    #------------------------------------------------        
    # show plot
    #------------------------------------------------                
    plt.show()
    
def plt_raster_rand(self, S, y_label):
    """
    Raster plot of spikes over time for second layer (Random layer)
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
    ax.set_title('R_stim:'+str(self.R_stim)+' R_Prop:'+str(self.R_StimProportion)+' StimStr:'+str(self.StimStrength))
    
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
    # rect_stim = patches.Rectangle((0,0), self.StimExposeTime/ms, y_lim, linewidth=0, facecolor='g', alpha = .2)        
    # rect_wm = patches.Rectangle((self.StimExposeTime/ms,0), (self.AttnTime/ms), y_lim, linewidth=0, facecolor='r', alpha = .2)        
    # ax.add_patch(rect_stim)
    # ax.add_patch(rect_wm)

    #------------------------------------------------        
    # if ping, show it...
    #------------------------------------------------  
    # if self.PingStimScale > 0:
    #     rect_ping = patches.Rectangle(((self.StimExposeTime+self.PingStimOnset)/ms,0), (self.PingExposeTime/ms), y_lim, linewidth=0, facecolor='k', alpha = .2)        
    #     ax.add_patch(rect_ping)
        
    # if plt_sensory:   
    #     plt.savefig('SensoryRaster-' + 'SetSize-' + str(self.SetSize) + '.jpg', dpi=600)
    # else:
    #     plt.savefig('RandomRaster-' + 'SetSize-' + str(self.SetSize) + '.jpg', dpi=600)

    #------------------------------------------------        
    # show plot
    #------------------------------------------------                
    plt.show()
    