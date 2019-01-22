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
    varpath = str(variable_scope+"/"+layer+"/"+parameter)
    varout = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, varpath)[0])
    return varout