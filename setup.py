# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 23:01:49 2018

@author: satya
"""
# Import dependencies

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv
import time

localtime = time.asctime( time.localtime(time.time()) )
print("Start time :", localtime)
 
samplen = 9000
batch_size = 1
generations = 10000

sample_length = 4800
kernel_stride = 2

# Architecture of Network --> get weights here

 
#--------------------------------------------------------------------------------------------------------------

# General Activation Function

def activation(layer_input,weights,bias):
    return tf.nn.sigmoid(tf.add(tf.matmul(layer_input,weights),bias))


#--------------------------------------------------------------------------------------------------------------

# Load data from csv
mag_dataset = []
choice = []

with open('../training_data.csv', 'rU') as f:  #opens PW file
    reader = csv.reader(f)
    mag_data = list(list(rec) for rec in csv.reader(f, delimiter=',')) #reads csv into a list of lists'


x_vals = np.array([x[0:sample_length] for x in mag_data])
choice = np.random.choice(len(x_vals), size = samplen)
x_vals = x_vals[np.array(choice)]

x_vals = x_vals.astype(np.float32)
x_vals = tf.keras.utils.normalize(x_vals,axis=-1,order=2)

train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
rem = np.array(list(set(range(len(x_vals))) - set(train_indices)))
validation_indices = np.random.choice(len(rem), round(len(rem)*0.5), replace=False)
test_indices = np.array(list(set(range(len(rem))) - set(validation_indices)))

# Use if dataset size is 1 
'''train_indices = [0]
test_indices = [0]
validation_indices = [0]'''

#y_vals = np.array([x[4800:4805] for x in mag_data]) # 4800:4805 for speed ; 4805:4813 for type if One hot encoded

y_vals = np.array([x[sample_length] for x in mag_data])
y_vals = np.array(y_vals[np.array(choice)])
y_vals = y_vals.astype(np.float)
y_vals = y_vals.astype(np.int32)

sess = tf.Session()

# Fix seed
seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

# 80:10:10 data split and normalization

x_vals_train = x_vals[train_indices]
x_vals_validation = x_vals[validation_indices]
x_vals_test = x_vals[test_indices]

y_vals_train = y_vals[train_indices]
y_vals_validation = y_vals[validation_indices]
y_vals_test = y_vals[test_indices]

y_vals_train = np.array(y_vals_train)
y_vals_validation = np.array(y_vals_validation)
y_vals_test = np.array(y_vals_test)

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)

#x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
#x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))
#x_vals_train = np.nan_to_num(x_vals_train)
#x_vals_test = np.nan_to_num(x_vals_test)

# Declare batch size and placeholders
    
print("Setup Complete")