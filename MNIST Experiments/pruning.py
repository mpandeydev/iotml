import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# Local libraries

import config
import models
import helpers

# Tensorflow Housekeeping

config_tf = tf.ConfigProto()
config_tf.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config_tf)

#Load Full precision model

inputs,labels,loss = models.model0()
saver = tf.train.Saver()
saver.restore(sess,config.full_precision_path)

helpers.get_layer_names()

conv2_weights = helpers.get_layer(sess,'conv_2','kernel')
conv2_bias = helpers.get_layer(sess,'conv_2','bias')

helpers.prune_filter(sess,'conv_1',5)
helpers.prune_filter(sess,'conv_2',5)
helpers.prune_filter(sess,'conv_3',5)

# Select Filters to Prune

## Collect layers

## Access filter values

# Prune Model

# Pruning Statistics