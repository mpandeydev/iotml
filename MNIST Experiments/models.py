import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Local Libraries
import config

batch_size  = config.batch_size
num_labels  = config.num_labels
img_h       = config.img_h
img_w       = config.img_w

num_labels = config.num_labels

def model0():
    
    model_params = config.model0_params
    
    inputs = tf.placeholder("float", [None, img_h,img_w,1])
    labels = tf.placeholder("int32", [None])
    
    conv_1 = tf.layers.conv2d(
                                inputs      =inputs,
                                filters     =model_params['conv_1']['filters'],
                                kernel_size =model_params['conv_1']['kernel_size'],
                                strides     =model_params['conv_1']['strides'],
                                padding     ="valid",
                                activation  =tf.nn.relu,
                                name        ="model0/conv_1"
                            )
    
    conv_2 = tf.layers.conv2d(
                                inputs      =conv_1,
                                filters     =model_params['conv_2']['filters'],
                                kernel_size =model_params['conv_2']['kernel_size'],
                                strides     =model_params['conv_2']['strides'],
                                padding     ="valid",
                                activation  =tf.nn.relu,
                                name        ="model0/conv_2"
                            )
    
    conv_3 = tf.layers.conv2d(
                                inputs      =conv_2,
                                filters     =model_params['conv_3']['filters'],
                                kernel_size =model_params['conv_3']['kernel_size'],
                                strides     =model_params['conv_3']['strides'],
                                padding     ="valid",
                                activation  =tf.nn.relu,
                                name        ="model0/conv_3"
                            )
    
    conv_flat = tf.contrib.layers.flatten(conv_3)
    
    logits = tf.layers.dense(
                                conv_flat,
                                num_labels, 
                                activation=tf.nn.relu,
                                name = "logits"
                            )
    
    probabilities = tf.nn.softmax(logits)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                  logits=logits)
    
    return inputs,labels,loss
    
    