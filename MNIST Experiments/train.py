import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# Local libraries

import config
import models

# Tensorflow Housekeeping

config_tf = tf.ConfigProto()
config_tf.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config_tf)

batch_size = config.batch_size
num_labels = config.num_labels
    
# Import Dataset

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Import Model

inputs,labels,loss = models.model0()
saver = tf.train.Saver()

optimizer = tf.train.AdagradOptimizer(config.step_size)

train_step = optimizer.minimize(loss)

# Initialize all variables
    
init = tf.global_variables_initializer()
sess.run(init)
localtime = time.asctime( time.localtime(time.time()) )
print("Initialized Variables:", localtime)

training_losses     = []
validation_losses   = []

# Train Model    

print("Training Started:", localtime)

for epoch in range(config.total_epochs):
    
    training_loss = 0.0
    iteration_count = 0
    
    for iteration in range(config.total_iterations):
        
        #Get random batch
        rand_index = np.random.choice(len(x_train), size = batch_size)
        rand_x = x_train[rand_index].reshape(batch_size,config.img_h,config.img_w,1)
        rand_y = y_train[rand_index]
        
        # Training step
        _, temp_loss = sess.run([train_step,loss], feed_dict={inputs : rand_x, labels : rand_y})
        
        training_loss += temp_loss
        iteration_count+= 1
        
    training_loss = training_loss/iteration_count    
    training_losses.append(training_loss)
    
    validation_loss = 0.0
    iteration_count = 0
    
    for sample_index in range(len(x_test)):
        
        #Get a test input
        test_input = x_test[sample_index].reshape(1,config.img_h,config.img_w,1)
        test_label = [y_test[sample_index]]
        
        # Cumulate Validation Loss
        single_val_loss = sess.run([loss], feed_dict={inputs : test_input, labels : test_label})
        validation_loss += single_val_loss[0] 
        iteration_count+= 1
        
    validation_loss = validation_loss/iteration_count    
    validation_losses.append(validation_loss)
    
    print('Epoch : ',epoch)
    print('Training Loss   : ',training_loss)
    print('Validation Loss : ',validation_loss)
    print()

print("Training Ended:", localtime)

# Training Statistics

plt.plot(training_losses,'b',label='Training Losses')
plt.plot(validation_losses,'r',label='Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Save full precision model

saver.save(sess,config.full_precision_path)
sess.close()