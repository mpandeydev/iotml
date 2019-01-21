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

optimizer = tf.train.AdagradOptimizer(config.step_size)

train_step = optimizer.minimize(loss)

# Initialize all variables
    
init = tf.global_variables_initializer()
sess.run(init)
localtime = time.asctime( time.localtime(time.time()) )
print("Initialized Variables:", localtime)

# Train Model    

for epoch in range(config.total_epochs):
    
    for iteration in range(config.total_iterations):
        
        #Get random batch
        rand_index = np.random.choice(len(x_train), size = batch_size)
        rand_x = x_train[rand_index].reshape(batch_size,config.img_h,config.img_w,1)
        rand_y = y_train[rand_index]
        
        # Training step
        _, temp_loss = sess.run([train_step,loss], feed_dict={inputs : rand_x, labels : rand_y} )
        
    print(epoch,' : ',temp_loss)

# Training Statistics

# Prune Model

# Pruning Statistics