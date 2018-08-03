# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 19:32:23 2018

@author: satya
"""
import tensorflow as tf
import numpy as np
import time

def import_npy():
    global x_vals,speeds,types
    bits = 8
    full_range = 2**bits
    tdata = np.load('../training_data_3d.npy')
    speeds = tdata[:,3,0]
    types = tdata[:,3,1]
    x_vals = tdata[:,0:3,:]
    max_value = (np.amax(x_vals)) 
    min_value = (np.amin(x_vals))
    quantum = np.float16((max_value-min_value)/(2**bits))
    quantas = []
    for i in range(0,full_range):
            quantas.append(np.float16(i*quantum)+min_value)
    
    flat_x_vals = x_vals.flatten()
    quantized_x_vals = []
    
    for i in flat_x_vals:
        #quantized_x_vals.append(np.float16(np.int8((((i-min_value)/(max_value-min_value))*2**bits)-2**(bits-1))*quantum))    
        intval = np.uint8(np.rint(((i-min_value)/(max_value-min_value))*full_range))
        quantized_x_vals.append(quantas[intval])
    
    quantized_x_vals = np.asarray(quantized_x_vals)
    quantized_x_vals = np.reshape(quantized_x_vals,np.shape(x_vals))
    return min_value, max_value,quantized_x_vals
    
minin,maxin,quantized_input = import_npy()