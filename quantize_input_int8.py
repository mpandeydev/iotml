# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 19:32:23 2018

@author: satya
"""
import numpy as np
import matplotlib.pyplot as plt

def import_npy():
    global x_vals,speeds,types
    bits = 8
    full_range = 2**bits
    
    tdata = np.load('../training_data_3d.npy')
    speeds = tdata[:,3,0]
    types = tdata[:,3,1]
    x_vals = tdata[:,0:3,:]
    max_value = (np.amax(x_vals)) 
    print("Max : ",max_value)
    min_value = (np.amin(x_vals))
    print("Min : ",min_value)
    
    flat_x_vals = x_vals.flatten()
    quantized_x_vals = []
    
    for i in flat_x_vals:
        # Round weights to nearest part
        if i<0:
            #print("This is i : ",i)
            intval = np.int8(np.rint((-i/min_value)*((full_range/2)-1)))
            #print("Neg : ",i," to ",intval)
        if i>0:
            #print("This is i : ",i)
            intval = np.int8(np.rint((i/max_value)*((full_range/2))))
            #print("Pos : ",i," to ",intval)
        if i==0:
            intval = 0
            print("Zero")
        # Assign integer associated with value        
        
        
        quantized_x_vals.append(intval)
    
    quantized_x_vals_flat = np.asarray(quantized_x_vals)
    quantized_x_vals = np.reshape(quantized_x_vals_flat,np.shape(x_vals))
    return min_value, max_value,quantized_x_vals,flat_x_vals,quantized_x_vals_flat
    
minin,maxin,quantized_input,flat_x,flat_q = import_npy()
tdata = np.load('../training_data_3d.npy')

def plot_values(index,axis,ranges):
    global quantized_input,tdata
    
    '''selmin = min(quantized_input[index,axis,ranges])
    selmax = max(quantized_input[index,axis,ranges])
    min_index = quantas.index(selmin)
    max_index = quantas.index(selmax)
    sel_quantas = quantas[min_index:max_index+1]
    length = len(ranges)'''
    
    plt.plot(tdata[index,axis,ranges], 'b')
    plt.title('Non Quantized Training Data')
    plt.show()

    plt.plot(quantized_input[index,axis,ranges], 'g')
    plt.title('Quantized Training Data')
    plt.show()
    
plot_values(403,2,range(0,1600))
np.save("../training_data_int8",quantized_input)