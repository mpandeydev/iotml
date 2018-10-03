# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:56:02 2018

@author: satya
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import glob, os  
import numpy as np
from sklearn import datasets

#User defined parameters

batch_size = 20

iterations = 5000

hidden_layer1_nodes = 2
hidden_layer2_nodes = 2
hidden_layer3_nodes = 2
hidden_layer4_nodes = 2

learning_rate = 0.0005

label_size = 8

datatype = tf.float16

# Function definitions

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return ((m-col_min) / (col_max - col_min))-0.5

path = "chunked"
magdata_csv = glob.glob(path + "/*.csv")

mag_data = []
for file in magdata_csv:
    with open(file, 'r') as f:
        file_data = []
        reader = csv.reader(f)
        for row in reader:
            if len(row)>0:
                file_data.append(round(float(row[0]),4))
        #if max(file_data)>10:
         #   mag_data.append(file_data)
        mag_data.append(file_data)

# mag_data is the entirety of training data
# [-2] is speed
    
training_data = []
labels = []
speed_labels = []
vehicle_labels = []
    
for sample in mag_data:
    training_data.append(sample[:-3])
    speed_labels.append(sample[-2])
    vehicle_labels.append(sample[-3])
    
# Sample size 1200 --> Shape = (1200,)
    
# convert to numpy arrays
training_data = np.asarray(training_data)
training_data = training_data.astype(np.float32)

raw_labels = vehicle_labels
new_labels = []
for label in raw_labels:
    temp_label = np.zeros(label_size)
    index = int(label)
    temp_label[index] = 1
    new_labels.append(temp_label)
labels = np.asarray(new_labels)

# Split to training and testing 80:20 --> For algo testing switch to 99%

train_indices = np.random.choice(len(training_data), round(len(training_data)*0.8), replace=False)
test_indices = np.array(list(set(range(len(training_data))) - set(train_indices)))

mag_data_train = training_data[train_indices]
mag_data_train = normalize_cols(mag_data_train)

mag_data_test = training_data[test_indices]
mag_data_test = normalize_cols(mag_data_test)

labels_train = labels[train_indices]
#labels_train = normalize_cols(labels_train)

labels_test = labels[test_indices]
#labels_test = normalize_cols(test_indices)

# Start TensorFlow code

sess = tf.Session()

# Declare batch size and placeholders

x_data = tf.placeholder(shape=[None,1200], dtype=datatype)
y_target = tf.placeholder(shape=[label_size,None,1], dtype=datatype)

# Architecture of Network --> get weights here

A1 = tf.Variable(tf.random_normal(shape=[1200,hidden_layer1_nodes],dtype=datatype))
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer1_nodes],dtype=datatype))

A2 = tf.Variable(tf.random_normal(shape=[hidden_layer1_nodes,hidden_layer2_nodes],dtype=datatype))
b2 = tf.Variable(tf.random_normal(shape=[hidden_layer2_nodes],dtype=datatype))

A3 = tf.Variable(tf.random_normal(shape=[hidden_layer2_nodes,hidden_layer3_nodes],dtype=datatype))
b3 = tf.Variable(tf.random_normal(shape=[hidden_layer3_nodes],dtype=datatype))

A4 = tf.Variable(tf.random_normal(shape=[hidden_layer3_nodes,hidden_layer4_nodes],dtype=datatype))
b4 = tf.Variable(tf.random_normal(shape=[hidden_layer4_nodes],dtype=datatype))

An = tf.Variable(tf.random_normal(shape=[hidden_layer4_nodes,label_size],dtype=datatype))
bn = tf.Variable(tf.random_normal(shape=[label_size],dtype=datatype))

# Layer outputs 

#hidden1_out = tf.nn.sigmoid(tf.add(tf.matmul(x_data,A1),b1))
#hidden2_out = tf.nn.sigmoid(tf.add(tf.matmul(hidden1_out,A2),b2))
#hidden3_out = tf.nn.sigmoid(tf.add(tf.matmul(hidden2_out,A3),b3))
#hidden4_out = tf.nn.sigmoid(tf.add(tf.matmul(hidden3_out,A4),b4))
#final_out = tf.nn.softmax(tf.add(tf.matmul(hidden4_out,An),bn))

hidden1_out =  tf.add(tf.matmul(x_data,A1),b1)
hidden2_out =  tf.add(tf.matmul(hidden1_out,A2),b2)
hidden3_out =  tf.add(tf.matmul(hidden2_out,A3),b3)
hidden4_out = tf.add(tf.matmul(hidden3_out,A4),b4)
final_out = tf.add(tf.matmul(hidden2_out,An),bn)

# Define Loss function

#loss = tf.reduce_mean(tf.square(y_target-final_out))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_out, labels=tf.transpose(y_target)))
#acc = tf.subtract(tf.transpose(y_target),tf.one_hot(tf.nn.top_k(final_out).indices, tf.shape(final_out)))

# Optimizer function

#my_opt = tf.train.GradientDescentOptimizer(0.0005)
my_opt = tf.train.AdamOptimizer(learning_rate)

train_step = my_opt.minimize(loss)

# Initialize all variables

init = tf.initialize_all_variables()
sess.run(init)

# Log vectors

loss_vec = []
test_loss = []
cum_loss = 0

# Train 

for i in range(iterations):
    
    # Get random batch
    
    rand_index = np.random.choice(len(mag_data_train), size = batch_size)
    #rand_index = np.random.choice(len(mag_data_train))
    rand_x = mag_data_train[rand_index]
    rand_y = np.transpose([labels_train[rand_index]])
    
    # Training step
    
    sess.run(train_step, feed_dict={x_data : rand_x, y_target:rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data : rand_x, y_target:rand_y} )
    loss_vec.append(np.sqrt(temp_loss))
    
    # Get testing loss
    
    test_temp_loss = sess.run(loss, feed_dict={x_data : mag_data_test, y_target:np.transpose([labels_test])})
    test_loss.append(np.sqrt(test_temp_loss))
    
    #print(temp_acc)
    
    cum_loss = 0
    cum_loss+=temp_loss
    # Print updates
    if (i+1)%100==0:
        cum_loss = cum_loss/100
        test_loss.append(np.sqrt(cum_loss))
        print('Generation: ' + str(i+1) + '. Loss = ' + str(cum_loss))
   
# Plot values   
plt.plot(loss_vec, 'k-', label='Train Loss')
#plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Final Loss : '+str(cum_loss))
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
#plt.savefig('../mag_plots/events/'+str(sel_vehicle)+'_'+str(sel_speed)+'_'+str(sel_sensor)+'.png')
plt.show()
sess.close()