import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# Local libraries

import config
import models

def get_layer_names():
    trainable_variables = [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
    for name in trainable_variables:
        print(name)
        
def get_layer(sess,layer,parameter):
    varpath = str(config.model+"/"+layer+"/"+parameter)
    varout = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, varpath)[0])
    return varout

def prune_filter(sess,layer,to_prune):
    
    kernels = get_layer(sess,layer,'kernel')
    bias = get_layer(sess,layer,'bias')
    
    k_height        = kernels.shape[0]
    k_width         = kernels.shape[1]
    num_channels    = kernels.shape[2]
    num_kernels     = kernels.shape[3]
    
    kernels = kernels.reshape(num_kernels,num_channels,k_height,k_width)
    
    print(layer)
    for kernel in kernels:
        print(np.sum(np.absolute(kernel)))
        
    print()
        
    return 0
    