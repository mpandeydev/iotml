import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

t_logit_size = 8
s_logit_size = 5

hidden_layer_nodes = 1000

hidden_layer_nodes_s = 1000
hidden_layer_nodes_2_s = 500
hidden_layer_nodes_3_s = s_logit_size

hidden_layer_nodes_t = 1000
hidden_layer_nodes_2_t = 500
hidden_layer_nodes_3_t = t_logit_size

step_size           = 0.05
 
samplen             = 9000
batch_size          = 200
input_channels      = 3
 
generations         = 15000

sample_length       = 1600

filter_width        = 8
kernel_stride       = 10
feature_maps       = 6

pool_sizes          = 4
pool_stride         = 4

conv_size           = 104

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


#--------------------------------------------------------------------------------------------------------------

def import_npy():
    global x_vals,speeds,types
    tdata = np.load('../training_data_3d.npy')
    speeds = tdata[:,3,0]
    types = tdata[:,3,1]
    x_vals = tdata[:,0:3,:]

def setup():
    global x_vals, y_vals, speeds, types, x_vals_train, x_vals_test, x_vals_validation, y_vals_test_t, y_vals_train_t, y_vals_validation_t,y_vals_test_s, y_vals_train_s, y_vals_validation_s,logit_size
    localtime = time.asctime( time.localtime(time.time()) )
    print("Start time :", localtime)  
    
    x_vals = np.array([x[0:sample_length] for x in x_vals])
    choice = np.random.choice(len(x_vals), size = samplen)
    x_vals = x_vals[np.array(choice)]
    
    x_vals = x_vals.astype(np.float32)
    x_vals = tf.keras.utils.normalize(x_vals,axis=-1,order=2)
    
    train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
    rem = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    validation_indices = np.random.choice(len(rem), round(len(rem)*0.5), replace=False)
    test_indices = np.array(list(set(range(len(rem))) - set(validation_indices)))
 
    y_vals_s = np.array([speeds[:]])
    y_vals_s = np.array(y_vals_s[0,np.array(choice)])
    y_vals_s = y_vals_s.astype(np.float)
    y_vals_s = y_vals_s.astype(np.int32)
       
    y_vals_t = np.array([types[:]])
    y_vals_t = np.array(y_vals_t[0,np.array(choice)])
    y_vals_t = y_vals_t.astype(np.float)
    y_vals_t = y_vals_t.astype(np.int32)
    
    localtime = time.asctime( time.localtime(time.time()) )
    print("Loaded Dataset :", localtime)
    
    # Fix seed
    seed = 2
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
    # 80:10:10 data split and normalization
    
    x_vals_train = x_vals[train_indices]
    x_vals_validation = x_vals[validation_indices]
    x_vals_test = x_vals[test_indices]
    
    y_vals_train_s = y_vals_s[train_indices]
    y_vals_validation_s = y_vals_s[validation_indices]
    y_vals_test_s = y_vals_s[test_indices]
    
    y_vals_train_t = y_vals_t[train_indices]
    y_vals_validation_t = y_vals_t[validation_indices]
    y_vals_test_t = y_vals_t[test_indices]
    
    
    # Declare batch size and placeholders
        
    localtime = time.asctime( time.localtime(time.time()) )
    print("Setup Complete :", localtime)

#*****************************************************************************
    
def run_network(fc , fc2):
    feature_maps = fc
    conv_size = fc2
    print("Parameter is : ",fc )
    
     # Placeholders
    x_data = tf.placeholder(shape=(None, sample_length,3), dtype=tf.float32)
    y_target_s = tf.placeholder(shape=(None), dtype=tf.int32) 
    y_target_t = tf.placeholder(shape=(None), dtype=tf.int32) 
    
    # Change functions from here on out if architecture changes
         
    # Set up Computation Graph 
    
    # Convolutional and Pooling Layers
    conv1d_f = tf.layers.conv1d(inputs=x_data,filters=feature_maps,kernel_size=filter_width,strides=kernel_stride,padding="valid",activation=tf.nn.relu)
    #conv1d_f = tf.layers.max_pooling1d(inputs=conv1d_f, pool_size=pool_sizes, strides=pool_stride)
    conv1d_flat = tf.reshape(conv1d_f, [-1, conv_size])
    
    # Fully Connected Layers
    
    fc_1 = tf.layers.dense(conv1d_flat,hidden_layer_nodes,activation=tf.nn.relu)
    
    #fc_1_s = tf.layers.dense(conv1d_flat,hidden_layer_nodes_s,activation=tf.nn.relu)
    
    fc_2_s = tf.layers.dense(fc_1,hidden_layer_nodes_2_s,activation=tf.nn.relu)
    fc_f_s = tf.layers.dense(fc_2_s,hidden_layer_nodes_3_s,activation=tf.nn.relu)
    fprob_s = tf.nn.softmax(fc_f_s, name=None)
    
    #fc_1_t = tf.layers.dense(conv1d_flat,hidden_layer_nodes_t,activation=tf.nn.relu)
    
    fc_2_t = tf.layers.dense(fc_1  ,hidden_layer_nodes_2_t,activation=tf.nn.relu)
    fc_f_t = tf.layers.dense(fc_2_t,hidden_layer_nodes_3_t,activation=tf.nn.relu)
    fprob_t = tf.nn.softmax(fc_f_t, name=None)
    
    localtime = time.asctime( time.localtime(time.time()) )
    print("Graph Defined :", localtime)
    
    #*****************************************************************************
    
    # Define Loss function
    
    speeds_loss = tf.losses.sparse_softmax_cross_entropy(labels=y_target_s,logits=fc_f_s)
    types_loss = tf.losses.sparse_softmax_cross_entropy(labels=y_target_t,logits=fc_f_t)
    loss = tf.add(speeds_loss,types_loss)
    
    # Optimizer function
    
    my_opt = tf.train.AdagradOptimizer(step_size)
    train_step = my_opt.minimize(loss)
    
    # Initialize all variables
    
    init = tf.global_variables_initializer()
    sess.run(init)
    localtime = time.asctime( time.localtime(time.time()) )
    print("Initialized Variables:", localtime)
    
    # Log vectors
    loss_vec = []
    test_loss = []
    
    # Speeds
    
    test_pred_s = []
    pred_vec_s = []
    success_rate_s = []
    successful_guesses_s = []
    test_rate_s = []    
    
    # Types
    
    test_pred_t = []
    pred_vec_t = []
    success_rate_t = []
    successful_guesses_t = []
    test_rate_t = []
    
    # Train 
    
    for i in range(generations):
        with tf.Graph().as_default():
        
            # Get random batch
            
            rand_index = np.random.choice(len(x_vals_train), size = batch_size)
            rand_x = [x_vals_train[rand_index]]
            
            rand_y_s = y_vals_train_s[rand_index]
            rand_y_t = y_vals_train_t[rand_index]
            
            # Training step
            
            sess.run(train_step, feed_dict={x_data : np.array([rand_x]).reshape((batch_size,sample_length,3)), y_target_s:rand_y_s, y_target_t:rand_y_t})
            temp_loss,pred_training_s,pred_training_t = sess.run([loss,fprob_s,fprob_t], feed_dict={x_data : np.array([rand_x]).reshape((batch_size,sample_length,input_channels)), y_target_s:rand_y_s, y_target_t:rand_y_t} )
            loss_vec.append(temp_loss)
            
            temp_loss,pred_training_s,pred_training_t = sess.run([loss,fprob_s,fprob_t], feed_dict={x_data : np.array([x_vals_train]).reshape((7200,sample_length,input_channels)), y_target_s:y_vals_train_s, y_target_t:y_vals_train_t} )
            
            guess_s = np.argmax(pred_training_s,axis=1)
            guess_t = np.argmax(pred_training_t,axis=1)
            
            pred_vec_s.append(guess_s)
            pred_vec_t.append(guess_t)
            
            correct_pred_s = np.sum(np.equal(guess_s,y_vals_train_s))
            correct_pred_t = np.sum(np.equal(guess_t,y_vals_train_t))
            
            
            successful_guesses_s.append(correct_pred_s)
            successful_guesses_t.append(correct_pred_t)
            
            success_rate_t.append(correct_pred_t/len(x_vals_train) )
            train_accuracy_t = round(correct_pred_t*100/len(x_vals_train),4)
            
            success_rate_s.append(correct_pred_s/len(x_vals_train) )
            train_accuracy_s = round(correct_pred_s*100/len(x_vals_train),4)
            
            # Get testing loss
   
            test_temp_loss,pred_testing_s,pred_testing_t = sess.run([loss,fprob_s,fprob_t], feed_dict={x_data : np.array([x_vals_test]).reshape((900,sample_length,input_channels)), y_target_s:y_vals_test_s, y_target_t:y_vals_test_t})
            
            test_loss.append(test_temp_loss)
            
            test_pred_s.append(pred_testing_s)
            test_guess = np.argmax(pred_testing_s,axis=1)
            test_correct_pred = np.sum(np.equal(test_guess,y_vals_test_s))
            test_rate_s.append(test_correct_pred/len(y_vals_test_s))
            test_accuracy_s = round(test_correct_pred*100/len(y_vals_test_s),4)
            
            test_pred_t.append(pred_testing_t)
            test_guess = np.argmax(pred_testing_t,axis=1)
            test_correct_pred = np.sum(np.equal(test_guess,y_vals_test_t))
            test_rate_t.append(test_correct_pred/len(y_vals_test_t))
            test_accuracy_t = round(test_correct_pred*100/len(y_vals_test_t),4)
            
            
            
            # Print updates
            if (i+1)%100==0:
                print('Generation: ' + str("{0:0=5d}".format(i+1))) 
                print('SPEEDS : Training Acc = ' + str((train_accuracy_s))+'. Test Acc = ' + str((test_accuracy_s))) 
                print('TYPES : Training Acc = ' + str((train_accuracy_t))+'. Test Acc = ' + str((test_accuracy_t))) 
                print('Loss = '  + str(round(temp_loss,4)))
                print()
        
    validation_loss,pred_validation_s,pred_validation_t = sess.run([loss,fprob_s,fprob_t], feed_dict={x_data : np.array([x_vals_validation]).reshape((900,sample_length,input_channels)), y_target_s:y_vals_validation_s,y_target_t:y_vals_validation_t})
    
    validation_guess_s = np.argmax(pred_validation_s,axis=1)
    validation_correct_pred_s = np.sum(np.equal(validation_guess_s,y_vals_validation_s))
    validation_accuracy_s = round(validation_correct_pred_s*100/len(y_vals_validation_s),4)
    
    validation_guess_t = np.argmax(pred_validation_t,axis=1)
    validation_correct_pred_t = np.sum(np.equal(validation_guess_t,y_vals_validation_t))
    validation_accuracy_t = round(validation_correct_pred_t*100/len(y_vals_validation_t),4)
    
    print("Validation Accuracies:")
    print("SPEEDS : ",str(validation_accuracy_s))
    print("TYPES : ",str(validation_accuracy_t))
    # Plot values
    localtime = time.asctime( time.localtime(time.time()) )
    print("End time :", localtime)
    
    
    plt.plot(loss_vec, 'k-', label='Train Loss')
    plt.plot(test_loss, 'r--', label='Test Loss')
    plt.title('Loss (MSE) per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
    
    plt.plot(success_rate_s, 'r--', label='Training Success rate')
    plt.plot(test_rate_s, 'b--', label='Test Success rate')
    plt.title('Speeds Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Success Rate')
    plt.legend(loc='upper right')
    plt.show()

    plt.plot(success_rate_t, 'r--', label='Training Success rate')
    plt.plot(test_rate_t, 'b--', label='Test Success rate')
    plt.title('Types Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Success Rate')
    plt.legend(loc='upper right')
    plt.show()
    
    sess.close()
  
'''import_npy()
setup("speeds")
run_network(15,180)
sess.close()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)'''

import_npy()
setup()
run_network(15,2400)
sess.close()