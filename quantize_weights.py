T# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:48:36 2018

@author: satya
"""

import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)

def import_weights():
    weights = np.load('weights_fullprecision.npy')
    
    max_values = [] 
    min_values = [] 
    
    for i in weights:
       max_values.append(np.amax(i)) 
       min_values.append(np.amin(i)) 
       
    #max_value = np.float16(np.amax(max_values))
    #min_value = np.float16(np.amax(min_values))
    return weights, max_values, min_values

def quantization_range(bits):
    full_range = 2**bits
    all_quantas = []
    for i in range(0,len(max_weight)):
        quantum = (max_weight[i] - min_weight[i])/full_range
        quantum = np.float16(quantum)
        quantas = []
        for i in range(0,full_range):
            quantas.append(np.float16(i*quantum))
        all_quantas.append(quantas)
        
    return all_quantas

def quantize_weights(weights,quantas):
    quantized_weights = []
    for i in range(0,len(weights)):
        to_reshape = []
        flat_weights = weights[i].flatten()
        cnt = 0
        curr_quantas = quantas[i]
        for j in flat_weights:
            quantized_value = min(curr_quantas, key=lambda x:abs(x-j))
            to_reshape.append(quantized_value)
            cnt+=1
        to_reshape = np.asarray(to_reshape)
        to_reshape = np.reshape(to_reshape,weights[i].shape)
        quantized_weights.append(to_reshape)
    return quantized_weights

#------------------------------------------------------------------------------


fp_weights, max_weight, min_weight = import_weights()
fweights = []
for i in fp_weights:
    fweights.append(i)

quantas = quantization_range(10)
#quantized_weights = np.load('weights_quantized_10bit_perlayer.npy')
#quantized_weights = quantize_weights(fp_weights,quantas)
quantized_weights = fweights

conv_layer_weights = tf.Variable(tf.convert_to_tensor(quantized_weights[0],dtype=tf.float32),name="conv_1d/kernel")
conv_bias_weights = tf.Variable(tf.convert_to_tensor(quantized_weights[1],dtype=tf.float32),name="conv_1d/bias")

fc_shared_layer_weights = tf.Variable(tf.convert_to_tensor(quantized_weights[2],dtype=tf.float32),name="fc_1/kernel")
fc_shared_bias_weights = tf.Variable(tf.convert_to_tensor(quantized_weights[3],dtype=tf.float32),name="fc_1/bias")

fc_speed_layer1_weights = tf.Variable(tf.convert_to_tensor(quantized_weights[4],dtype=tf.float32),name="fc_2_s/kernel")
fc_speed_bias1_weights = tf.Variable(tf.convert_to_tensor(quantized_weights[5],dtype=tf.float32),name="fc_2_s/bias")

fc_speed_layer2_weights = tf.Variable(tf.convert_to_tensor(quantized_weights[6],dtype=tf.float32),name="fc_f_s/kernel")
fc_speed_bias2_weights = tf.Variable(tf.convert_to_tensor(quantized_weights[7],dtype=tf.float32),name="fc_f_s/bias")

fc_types_layer1_weights = tf.Variable(tf.convert_to_tensor(quantized_weights[8],dtype=tf.float32),name="fc_2_t/kernel")
fc_types_bias1_weights = tf.Variable(tf.convert_to_tensor(quantized_weights[9],dtype=tf.float32),name="fc_2_t/bias")

fc_types_layer2_weights = tf.Variable(tf.convert_to_tensor(quantized_weights[10],dtype=tf.float32),name="fc_f_t/kernel")
fc_types_bias2_weights = tf.Variable(tf.convert_to_tensor(quantized_weights[11],dtype=tf.float32),name="fc_f_t/bias")

init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
saver.save(sess,"../quantized_model_perlayer_full.ckpt")
