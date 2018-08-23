import tensorflow as tf
import tfgraphviz as tfg
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import csv

import os

#tf.enable_eager_execution()

os.environ["PATH"] += os.pathsep + 'D:/CMU_Summer18/graphviz-2.38/release/bin/'

precision = tf.float16
logit_size = 8

hidden_layer_nodes = 900
hidden_layer_nodes_2 = 500
#hidden_layer_nodes_3 = 100

hidden_layer_nodes_f = logit_size # Must be the same as output logits

step_size           = 0.2

 
samplen             = 9000
batch_size          = 200
input_channels      = 3
 
generations         = 1000
loss_limit          = 0.02

sample_length       = 1600

filter_width        = 8
kernel_stride       = 4
feature_maps        = 12

filter_width_2        = 8
kernel_stride_2       = 4
feature_maps_2        = 8

filter_width_3        = 6
kernel_stride_3       = 4
feature_maps_3        = 4

pool_sizes          = 4
pool_stride         = 4

conv_size           = 104
quantize_train      = 1

quantize_on     = True

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)

x_vals = []
y_vals = []

speeds = []
types = []

x_vals_train = []
x_vals_validation = []
x_vals_test = []
    
y_vals_train = []
y_vals_validation = []
y_vals_test = []

#--------------------------------------------------------------------------------------------------------------

# General Activation Function

def activation(layer_input,weights,bias):
    return tf.nn.relu(tf.add(tf.matmul(layer_input,weights),bias))

# Quantization Function
    
def quantize_tensor(input_tensor,bits_q):

    #if(quantize_on == False):
        #return input_tensor
    
    bits = bits_q
    full_range = 2**bits
    converted_array = []
    
    input_array = sess.run(input_tensor)
    max_value = np.float16(np.amax(input_array)) 
    min_value = np.float16(np.amin(input_array))
    quantum = np.float16((max_value-min_value)/full_range)
    quantas = []
    for i in range(0,full_range):
            quantas.append(np.float16(i*quantum)+min_value)
    input_array_flat = input_array.flatten()
    
    for i in input_array_flat:
        intval = np.uint8(np.rint(((i-min_value)/(max_value-min_value))*full_range))
        converted_array.append(quantas[intval])
    
    converted_array = np.asarray(converted_array)
    converted_array = np.reshape(converted_array,np.shape(input_array))
    converted_array_t = tf.convert_to_tensor(converted_array,dtype=precision)
    assign_op = tf.assign(input_tensor,converted_array_t)
    sess.run(assign_op)
    return converted_array

def op_convert_to_int8(i):
    quantized_tensor = []
    zero = tf.cast(0,tf.float32)
    one = tf.cast(1,tf.float32)
    # Round weights to nearest part
    
    def neg(): 
        quantized_tensor.append(0)
        return 0
    def pos(): 
        quantized_tensor.append(1)
        return 1
    r = tf.cond(tf.greater(zero, i[0]), neg, pos)
    return tf.cast(quantized_tensor,tf.float32)
        
def convert_tensor_to_int8(input_tensor):
    full_range = 256
    input_array = input_tensor
    
    max_value = tf.reduce_max(input_array)
    min_value = tf.reduce_min(input_array)
    
    flat_x_vals = tf.layers.flatten(input_array)
    quantized_tensor = []
    
    quantized_tensor = tf.map_fn(op_convert_to_int8,flat_x_vals)
    
    #converted_array = sess.run(quantized_tensor)
    #print(converted_array)
    converted_array_t = tf.cast(quantized_tensor,dtype=np.float16)
    #converted_array_t = tf.reshape(converted_array_t, tf.shape(input_tensor))
    #assign_op = tf.assign(input_tensor,converted_array_t)
    #sess.run(assign_op)
    return converted_array_t   


#--------------------------------------------------------------------------------------------------------------

def import_npy():
    global x_vals,speeds,types
    tdata = np.load('../../training_data_3d.npy')
    print(np.size(tdata))
    speeds = tdata[:,3,0]
    types = tdata[:,3,1]
    x_vals = tdata[:,0:3,:]
    
    x_vals = tf.keras.utils.normalize(x_vals,axis=-1,order=2)
    
    # Zero means
    #'''
    original_shape = x_vals.shape
    x_vals = x_vals.reshape(-1,1600)
    means = np.mean(x_vals,dtype=np.float16,axis=(1))
    means = means.reshape(27024,1)
    x_vals = x_vals-means
    x_vals = x_vals.reshape(original_shape)
    #'''
    
    return x_vals

#*****************************************************************************
def print_trainable_variables(train_variables):
    for v in train_variables:
        print(v.name)

with tf.variable_scope("foo",reuse=tf.AUTO_REUSE):
    norm_data = import_npy()
    
    #sess.close()
