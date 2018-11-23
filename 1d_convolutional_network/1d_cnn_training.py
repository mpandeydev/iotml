import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import csv

import os
 
selected_data = 'types'

precision = tf.float16
precision_np = np.float32
logit_size = 8

quantization = True

hidden_layer_nodes = 900
hidden_layer_nodes_2 = 500

hidden_layer_nodes_f = logit_size # Must be the same as output logits

step_size           = 0.2

 
samplen             = 9000
batch_size          = 200
input_channels      = 3
 
generations         = 500
loss_limit          = 0.02

sample_length       = 1600

filter_width        = 8
kernel_stride       = 4
feature_maps        = 12

filter_width_2        = 8
kernel_stride_2       = 4
feature_maps_2        = 8

filter_width_3        = 6
kernel_stride_3       = 4
feature_maps_3        = 4

pool_sizes          = 4
pool_stride         = 4

conv_size           = 104
quantize_train      = 1

quantize_on     = True

# Configure TF session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)

# Initialize empty lists to hold data
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

def plot_histogram(input_tensor):
    flat_inputs = input_tensor.flatten()
    plt.hist(flat_inputs,bins = 16)
    plt.show()
    
# General Activation Function

def activation(layer_input,weights,bias):
    return tf.nn.relu(tf.add(tf.matmul(layer_input,weights),bias))

# Quantization Function
    
def quantize_tensor(input_tensor,bits_q):
    
    bits = bits_q
    full_range = 2**bits
    converted_array = []   
    input_array = sess.run(input_tensor)
    max_value = np.float16(np.amax(input_array)) 
    min_value = np.float16(np.amin(input_array))
    
    ## Quantize range to a resolution given by full_range
    
    quantum = np.float16((max_value-min_value)/full_range) #Quantum is a value that 1 bit represents
    quantas = []
    for i in range(0,full_range):
            quantas.append(np.float16(i*quantum)+min_value)
    input_array_flat = input_array.flatten()
    
    for i in input_array_flat:
        intval = np.uint8(np.rint(((i-min_value)/(max_value-min_value))*full_range))
        converted_array.append(quantas[intval])
    
    converted_array = np.asarray(converted_array)
    converted_array = np.reshape(converted_array,np.shape(input_array))
    converted_array_t = tf.convert_to_tensor(converted_array,dtype=precision)
    assign_op = tf.assign(input_tensor,converted_array_t)
    sess.run(assign_op)
    return converted_array

def op_convert_to_int8(i):
    quantized_tensor = []
    zero = tf.cast(0,precision)
    
    # Round weights to nearest part
    
    def neg(): 
        quantized_tensor.append(0)
        return 0
    
    def pos(): 
        quantized_tensor.append(1)
        return 1
    
    return tf.cast(quantized_tensor,precision)
        
def convert_tensor_to_int8(input_tensor):
    
    input_array = input_tensor
    
    flat_x_vals = tf.layers.flatten(input_array)
    quantized_tensor = []
    
    quantized_tensor = tf.map_fn(op_convert_to_int8,flat_x_vals)
    
    converted_array_t = tf.cast(quantized_tensor,dtype=precision)

    return converted_array_t   


#--------------------------------------------------------------------------------------------------------------

def import_npy():
    global x_vals,speeds,types
    tdata = np.load('../../datasets/training_data_3d.npy')
    
    speeds  = tdata[:,3,0]
    types   = tdata[:,3,1]
    x_vals  = tdata[:,0:3,:].astype(precision_np)
    
    
    # Zero means

    original_shape = x_vals.shape
    x_vals = x_vals.reshape(-1,1600)
    means = np.mean(x_vals,dtype=precision_np,axis=(1))
    means = means.reshape(27024,1)
    x_vals = x_vals-means
    x_vals = x_vals.reshape(original_shape)
    
    
def setup(dataset):
    global x_vals, y_vals
    global speeds, types
    global x_vals_train, x_vals_test, x_vals_validation 
    global y_vals_test, y_vals_train, y_vals_validation
    global logit_size
    
    localtime = time.asctime( time.localtime(time.time()) )
    print("Start time :", localtime)  
    
    x_vals = np.array([x[0:sample_length] for x in x_vals])
    choice = np.random.choice(len(x_vals), size = samplen)
    x_vals = x_vals[np.array(choice)]
    
    train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), 
                                     replace=False)
    rem = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    validation_indices = np.random.choice(len(rem), round(len(rem)*0.5), 
                                          replace=False)
    test_indices = np.array(list(set(range(len(rem))) - set(validation_indices)))
    
    # Change for datasets
    
    if(dataset=="speeds"):
        logit_size = 5
        y_vals = np.array([speeds[:]])
        
    if(dataset=="types"):
        logit_size = 8
        y_vals = np.array([types[:]])
        
    y_vals = np.array(y_vals[0,np.array(choice)])
    y_vals = y_vals.astype(np.int32)
    
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
    
    y_vals_train = y_vals[train_indices]
    y_vals_validation = y_vals[validation_indices]
    y_vals_test = y_vals[test_indices]
    
    # Declare batch size and placeholders
        
    localtime = time.asctime( time.localtime(time.time()) )
    print("Setup Complete :", localtime)

#*****************************************************************************
def print_trainable_variables(train_variables):
    for v in train_variables:
        print(v.name)
        
def run_network():
    global saver,quantization
    
    # Placeholders
    
    x_data = tf.placeholder(shape=(None, sample_length,3), dtype=precision)
    y_target = tf.placeholder(shape=(None), dtype=tf.int32)
    
    
    # Change functions from here on out if architecture changes
    
    kernel_init = tf.glorot_uniform_initializer(
                                                seed=None,
                                                dtype=precision)
         
# Set up Computation Graph 
    
    # Convolution Layer 1

    conv1d_1 = tf.layers.conv1d(
                                inputs=x_data,
                                filters=feature_maps,
                                kernel_size=filter_width,
                                strides=kernel_stride,
                                padding="valid",
                                activation=tf.nn.relu,
                                kernel_initializer = kernel_init,
                                name="conv1d_1"
                                )
    conv1d_1_cast = tf.cast(conv1d_1,tf.float32)
    conv1d_1_norm = tf.layers.batch_normalization(conv1d_1_cast, 
                                                  training = True, 
                                                  fused=False, 
                                                  name="bn1")
    conv1d_1_norm16 = tf.cast(conv1d_1_norm,precision)
    
    # Convolution Layer 2
    
    conv1d_2 = tf.layers.conv1d(
                                inputs=conv1d_1_norm16,
                                filters=feature_maps_2,
                                kernel_size=filter_width_2,
                                strides=kernel_stride_2,
                                padding="valid",
                                activation=tf.nn.relu,
                                kernel_initializer = kernel_init,
                                name="conv1d_2"
                                )
    conv1d_2_cast = tf.cast(conv1d_2,tf.float32)
    conv1d_2_norm = tf.layers.batch_normalization(conv1d_2_cast, 
                                                  training = True, 
                                                  fused=False, 
                                                  name = "bn2")
    conv1d_2_norm16 = tf.cast(conv1d_2_norm,precision)
    
    # Final Convolution Layer 
    
    conv1d_f = tf.layers.conv1d(
                                inputs=conv1d_2_norm16,
                                filters=feature_maps_3,
                                kernel_size=filter_width_3,
                                strides=kernel_stride_3,
                                padding="valid",
                                activation=tf.nn.relu,
                                kernel_initializer = kernel_init,
                                name="conv1d_f"
                                )
    conv1d_3_cast = tf.cast(conv1d_f,tf.float32)
    conv1d_3_norm = tf.layers.batch_normalization(conv1d_3_cast, 
                                                  training = True, 
                                                  fused=False, 
                                                  name = "bn3")
    conv1d_3_norm16 = tf.cast(conv1d_3_norm,precision)
    
    # Flatten Feature Maps for FC Layers
    
    conv1d_flat = tf.contrib.layers.flatten(conv1d_3_norm16)
    
    # Fully Connected Layers
    
    
    fc_f = tf.layers.dense(
                            conv1d_flat,
                            hidden_layer_nodes_f, 
                            activation=tf.nn.relu,
                            name = "logits"
                            )
    fc_f_16 = tf.cast(fc_f,precision)  
    fprob = tf.nn.softmax(fc_f_16)
    
    localtime = time.asctime( time.localtime(time.time()) )
    print("Graph Defined :", localtime)
    
#*****************************************************************************
    
    saver = tf.train.Saver()
    
# Define Loss function
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_target,
                                                  logits=fc_f_16)
    
# Optimizer function
    
    my_opt = tf.train.AdagradOptimizer(step_size)
    
    # First quantization cycle
    
    # If needed, remove variables by name from train_vars using :
    # train_vars.remove(tf.get_variable('variable_name',[variable_shape_tuple]))
    
    train_vars = tf.trainable_variables() 
    train_step = my_opt.minimize(loss,var_list=train_vars)
    
# Initialize all variables
    
    init = tf.global_variables_initializer()
    sess.run(init)
    localtime = time.asctime( time.localtime(time.time()) )
    print("Initialized Variables:", localtime)
    
    # Log vectors
    
    loss_vec = []
    test_loss = []
    test_pred = []

    success_rate = []
    successful_guesses = []
    test_rate = []
    qweights = []
    weights = []
    
    temp_loss_t = 1
    i = 0
    
    # Train 
    
    qweights = []
    weights = []
    order = [conv1d_1,conv1d_2, conv1d_f,fc_f]
    
    conv1d_f_over_time = []
    
    for h in range(quantize_train):
        print( )
        print("Quantize Train Loop iteration : ",h)
        i = 0
        
        for i in range(generations):
        # if stopping critireon is loss     
        # while(temp_loss_t>loss_limit):
        
            i+=1
            dataset_size = len(x_vals)    
            num_iters = dataset_size//batch_size
    
            for iter in range(num_iters):
    
                #Get random batch
                rand_index = np.random.choice(len(x_vals_train), 
                                              size = batch_size)
                rand_x = [x_vals_train[rand_index]]
                rand_y = y_vals_train[rand_index]
                
                # Training step
                _, temp_loss,get_pred = sess.run([train_step,loss,fprob], 
                                                 feed_dict={x_data : np.array([rand_x]).reshape(batch_size,sample_length,input_channels), 
                                                            y_target:rand_y} )
                
            loss_vec.append(temp_loss)
            
            #After the epoch is done, calculate loss, training, validation, test accuracy
            #At the end the following are plotted loss_vec, test_loss, success_rate, test_rate
            
            #Training related loss and prediction
            
            temp_loss_t,get_pred = sess.run([loss,fprob], 
                                            feed_dict={x_data : np.array([x_vals_train]).reshape((7200,sample_length,input_channels)), 
                                                       y_target:y_vals_train} )
            loss_vec.append(temp_loss_t)
            guess = np.argmax(get_pred,axis=1)
            correct_pred = np.sum(np.equal(guess,y_vals_train))
            successful_guesses.append(correct_pred)
            success_rate.append(correct_pred/len(x_vals_train) )
            train_accuracy = round(correct_pred*100/len(x_vals_train),4)
            
            # Get testing loss
            
            test_temp_loss,predict = sess.run([loss,fprob], 
                                              feed_dict={x_data : np.array([x_vals_test]).reshape((900,sample_length,input_channels)), 
                                                         y_target:y_vals_test})
            test_loss.append(test_temp_loss)
            test_pred.append(predict)
            test_guess = np.argmax(predict,axis=1)
            test_correct_pred = np.sum(np.equal(test_guess,y_vals_test))
            test_rate.append(test_correct_pred/len(y_vals_test))
            test_accuracy = round(test_correct_pred*100/len(y_vals_test),4)
            
            # Print updates
            
            if (i+1)%50==0:
                print('Generation: ' + str("{0:0=5d}".format(i+1)) + '. Training Acc = ' + str((train_accuracy))+'. Test Acc = ' + str((test_accuracy))+ '. Loss = '  + str(round(temp_loss_t,4)))
                qweights = []
                weights = []
                
                
                with tf.variable_scope("foo",reuse=tf.AUTO_REUSE):
                    weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(conv1d_1.name)[0] + '/kernel:0')))
                    weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(conv1d_2.name)[0] + '/kernel:0')))
                    weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(conv1d_f.name)[0] + '/kernel:0')))
                    weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_f.name)[0] + '/kernel:0')))
                    
                    conv_f_weights = sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(conv1d_f.name)[0] + '/kernel:0'))
                    conv1d_f_over_time.append(np.reshape(np.array(conv_f_weights),(-1)))
                    #plot_histogram(conv_f_weights)
                    
                    
                if(quantization):
                    print("Quantizing...")
                    for h in range(len(order)):
                        qweights.append(quantize_tensor(tf.get_default_graph().get_tensor_by_name(os.path.split(order[h].name)[0] + '/kernel:0'),8))
                        quantize_tensor(tf.get_default_graph().get_tensor_by_name(os.path.split(order[h].name)[0] + '/bias:0'),8)
    
            validation_loss,validation_predict = sess.run([loss,fprob], 
                                                          feed_dict={x_data : np.array([x_vals_validation]).reshape((900,sample_length,input_channels)), 
                                                                     y_target:y_vals_validation})
            validation_guess = np.argmax(validation_predict,axis=1)
            validation_correct_pred = np.sum(np.equal(validation_guess,y_vals_validation))
            validation_accuracy = round(validation_correct_pred*100/len(y_vals_validation),4)
            
            # Plot values
            
            localtime = time.asctime( time.localtime(time.time()) )
            
        print('Generation: ' + str("{0:0=5d}".format(i+1)) + '. Training Acc = ' + str((train_accuracy))+'. Test Acc = ' + str((test_accuracy))+ '. Loss = '  + str(round(temp_loss,4)))
        print("Validation Accuracy = "+str(validation_accuracy))
        print()
                      
        #Training related loss and prediction
        
        temp_loss,get_pred = sess.run([loss,fprob], 
                                      feed_dict={x_data : np.array([x_vals_train]).reshape((7200,sample_length,input_channels)), 
                                                 y_target:y_vals_train} )
        loss_vec.append(temp_loss)
        guess = np.argmax(get_pred,axis=1)
        correct_pred = np.sum(np.equal(guess,y_vals_train))
        successful_guesses.append(correct_pred)
        success_rate.append(correct_pred/len(x_vals_train) )
        train_accuracy = round(correct_pred*100/len(x_vals_train),4)
        
        # Get testing loss
        
        test_temp_loss,predict = sess.run([loss,fprob], 
                                          feed_dict={x_data : np.array([x_vals_test]).reshape((900,sample_length,input_channels)), 
                                                     y_target:y_vals_test})
        test_loss.append(test_temp_loss)
        test_pred.append(predict)
        test_guess = np.argmax(predict,axis=1)
        test_correct_pred = np.sum(np.equal(test_guess,y_vals_test))
        test_rate.append(test_correct_pred/len(y_vals_test))
        test_accuracy = round(test_correct_pred*100/len(y_vals_test),4)    
    
        validation_loss,validation_predict = sess.run([loss,fprob], 
                                                      feed_dict={x_data : np.array([x_vals_validation]).reshape((900,sample_length,input_channels)), 
                                                                 y_target:y_vals_validation})
        validation_guess = np.argmax(validation_predict,axis=1)
        validation_correct_pred = np.sum(np.equal(validation_guess,y_vals_validation))
        validation_accuracy = round(validation_correct_pred*100/len(y_vals_validation),4)
        
        print("QUANTIZED")
        print('Generation: ' + str("{0:0=5d}".format(i+1)) + '. Training Acc = ' + str((train_accuracy))+'. Test Acc = ' + str((test_accuracy))+ '. Loss = '  + str(round(temp_loss,4)))
        print("Validation Accuracy = "+str(validation_accuracy))
        print("___________________________________________________________")
    
    plt.plot(loss_vec, 'k-', label='Train Loss')
    plt.plot(test_loss, 'r--', label='Test Loss')
    plt.title('Loss (MSE) per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
    
    plt.plot(success_rate, 'r--', label='Training Success rate')
    plt.plot(test_rate, 'b--', label='Test Success rate')
    plt.title('Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Success Rate')
    plt.legend(loc='upper right')
    plt.show()
    
    return weights,qweights,np.array(conv1d_f_over_time)

#______________________________________________________________________________

def save_network(save_path):
    return saver.save(sess,save_path)
    
with tf.variable_scope("foo",reuse=tf.AUTO_REUSE):
    import_npy()
    setup(selected_data)
    weights,weights_q,cf_over_time = run_network()
    variables_names = [v.name for v in tf.trainable_variables()]
    for i in weights:
        plot_histogram(i)
        print(len(np.unique(i)))
    print("QUANTIZED")
    for i in weights_q:
        plot_histogram(i)
        print(len(np.unique(i)))
    save_network("../trained_models/quantized/quantized_8_types.ckpt")
    sess.close()
    