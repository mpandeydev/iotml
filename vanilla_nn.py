# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 23:01:49 2018

@author: satya
"""
# Import dependencies

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import csv
 
samplen = 9000
batch_size = 1000
generations = 100000

sample_length = 4800

# Architecture of Network --> get weights here

hidden_layer_nodes = 200
hidden_layer_nodes_2 = 150
hidden_layer_nodes_3 = 100
hidden_layer_nodes_4 = 75
hidden_layer_nodes_5 = 50
hidden_layer_nodes_6 = 20
hidden_layer_nodes_7 = 10

hidden_layer_nodes_f = 5

step_size = 0.005

#--------------------------------------------------------------------------------------------------------------

# Load data from csv
mag_dataset = []
choice = []

with open('training_data.csv', 'rU') as f:  #opens PW file
    reader = csv.reader(f)
    mag_data = list(list(rec) for rec in csv.reader(f, delimiter=',')) #reads csv into a list of lists


x_vals = np.array([x[0:sample_length] for x in mag_data])

'''choice = np.random.choice(len(x_vals), size = samplen)
x_vals = x_vals[np.array(choice)]'''

x_vals = x_vals.astype(np.float)
x_vals = tf.keras.utils.normalize(x_vals,axis=-1,order=2)

train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
rem = np.array(list(set(range(len(x_vals))) - set(train_indices)))
validation_indices = np.random.choice(len(rem), round(len(rem)*0.5), replace=False)
test_indices = np.array(list(set(range(len(rem))) - set(validation_indices)))

#y_vals = np.array([x[4800:4805] for x in mag_data]) # 4800:4805 for speed ; 4805:4813 for type if One hot encoded

y_vals = np.array([x[sample_length] for x in mag_data])
#y_vals = np.array(y_vals[np.array(choice)])
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

x_data = tf.placeholder(shape=[None, sample_length], dtype=tf.float32)
y_target = tf.placeholder(shape=[None], dtype=tf.int32)

#*****************************************************************************
# Change functions from here on out if architecture changes

'''filterz = tf.random_normal(shape=[3,1,1], dtype=tf.float32)
filterz = tf.expand_dims(filterz, axis = 2)
#conv2 = tf.nn.conv1d(x_data,filterz, stride=2, padding="VALID")
conv2 = tf.layers.conv1d(inputs=x_data,filters=1,kernel_size=3,strides=2,use_bias='True',padding="VALID")
dense = tf.layers.dense(inputs=conv2, units=1024, activation=tf.nn.relu)
A1 = tf.Variable(tf.random_normal(shape=[conv2,hidden_layer_nodes]))'''

A1 = tf.Variable(tf.random_normal(shape=[sample_length,hidden_layer_nodes]))
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))

A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,hidden_layer_nodes_2]))
b2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_2]))

A3 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_2,hidden_layer_nodes_3]))
b3 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_3]))

A4 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_3,hidden_layer_nodes_4]))
b4 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_4]))

A5 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_4,hidden_layer_nodes_5]))
b5 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_5]))

A6 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_5,hidden_layer_nodes_6]))
b6 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_6]))

A7 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_6,hidden_layer_nodes_7]))
b7 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_7]))

Af = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_7,hidden_layer_nodes_f]))
bf = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_f]))

# Layer outputs 

hidden_out = tf.nn.relu(tf.add(tf.matmul(x_data,A1),b1))
hidden_out_2 = tf.nn.relu(tf.add(tf.matmul(hidden_out,A2),b2))
hidden_out_3 = tf.nn.relu(tf.add(tf.matmul(hidden_out_2,A3),b3))
hidden_out_4 = tf.nn.relu(tf.add(tf.matmul(hidden_out_3,A4),b4))
hidden_out_5 = tf.nn.relu(tf.add(tf.matmul(hidden_out_4,A5),b5))
hidden_out_6 = tf.nn.relu(tf.add(tf.matmul(hidden_out_5,A6),b6))
hidden_out_7 = tf.nn.relu(tf.add(tf.matmul(hidden_out_6,A7),b7))
final_out = tf.nn.relu(tf.add(tf.matmul(hidden_out_7,Af),bf))
fprob = tf.nn.softmax(final_out, name=None)
 

#*****************************************************************************

# Define Loss function

loss = tf.losses.sparse_softmax_cross_entropy(labels=y_target,logits=final_out)
#loss = tf.losses.softmax_cross_entropy(onehot_labels=y_target,logits=final_out)

# Optimizer function

my_opt = tf.train.AdagradOptimizer(step_size)
#my_opt = tf.train.AdagradOptimizer(step_size)
train_step = my_opt.minimize(loss)

# Initialize all variables

init = tf.initialize_all_variables()
sess.run(init)

# Log vectors

loss_vec = []
test_loss = []
test_pred = []
pred_vec = []
success_rate = []
successful_guesses = []
# Train 

for i in range(generations):
    
    # Get random batch
    
    rand_index = np.random.choice(len(x_vals_train), size = batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = y_vals_train[rand_index]
    
    # Training step
    
    sess.run(train_step, feed_dict={x_data : rand_x, y_target:rand_y})
    temp_loss,get_pred = sess.run([loss,fprob], feed_dict={x_data : rand_x, y_target:rand_y} )
    loss_vec.append(temp_loss)
    guess = np.argmax(get_pred,axis=1)
    
    pred_vec.append(guess)
    correct_pred = np.sum(np.equal(guess,rand_y))
    successful_guesses.append(correct_pred)
    success_rate.append(correct_pred/batch_size)
    
    # Get testing loss
    
    test_temp_loss,predict = sess.run([loss,fprob], feed_dict={x_data : x_vals_test, y_target:y_vals_test})
    test_loss.append(test_temp_loss)
    test_pred.append(predict)
    test_guess = np.argmax(predict,axis=1)
    test_correct_pred = np.sum(np.equal(test_guess,y_vals_test))
    test_accuracy = round(test_correct_pred*100/len(y_vals_test),4)
    
    # Print updates
    if (i+1)%100==0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str((temp_loss))+'. Test Loss = ' + str((test_temp_loss))+'. Test Accuracy = ' + str((test_accuracy)))
        #print('Generation: ' + str(i+1) + '. Loss = ' + str((temp_loss))+". Accuracy "+str(correct_pred*100/batch_size)+"%")
        
# Plot values
        
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

plt.plot(success_rate, 'r--', label='Test Loss')
plt.title('Accuracy')
plt.xlabel('Generation')
plt.ylabel('Success Rate')
plt.legend(loc='upper right')
plt.show()