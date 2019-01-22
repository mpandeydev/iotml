import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import csv

import os

precision = tf.float16
precision_np = np.float32
logit_size = 8

quantization = True

hidden_layer_nodes = 900
hidden_layer_nodes_2 = 500

hidden_layer_nodes_speeds = 5 # Must be the same as output logits
hidden_layer_nodes_types = 8

step_size           = 0.2

 
samplen             = 9000
batch_size          = 200
input_channels      = 3
 
generations         = 100
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
    
    speeds  = np.array(tdata[:,3,0])
    types   = np.array(tdata[:,3,1])
    x_vals  = tdata[:,0:3,:].astype(precision_np)
    
    
    # Zero means

    original_shape = x_vals.shape
    x_vals = x_vals.reshape(-1,1600)
    means = np.mean(x_vals,dtype=precision_np,axis=(1))
    means = means.reshape(27024,1)
    x_vals = x_vals-means
    x_vals = x_vals.reshape(original_shape)
    
    
def setup():
    global x_vals, y_vals
    global speeds, types
    global x_vals_train, x_vals_test, x_vals_validation 
    global speed_labels_train, speed_labels_test, speed_labels_validation
    global type_labels_train, type_labels_test, type_labels_validation
    global logit_size
    global speed_labels,type_labels
    
    localtime = time.asctime( time.localtime(time.time()) )
    print("Start time :", localtime)  
    
    x_vals = np.array([x[0:sample_length] for x in x_vals])
    choice = np.random.choice(len(x_vals), size = samplen)
    x_vals = x_vals[np.array(choice)]
    
    train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
    rem = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    
    validation_indices = np.random.choice(len(rem), round(len(rem)*0.5),replace=False)
    test_indices = np.array(list(set(range(len(rem))) - set(validation_indices)))
        
    speed_labels = np.array(speeds[np.array(choice)]).astype(np.int32)
    type_labels = np.array(types[np.array(choice)]).astype(np.int32)
    
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
    
    speed_labels_train = speed_labels[train_indices]
    speed_labels_validation = speed_labels[validation_indices]
    speed_labels_test = speed_labels[test_indices]
    
    type_labels_train = type_labels[train_indices]
    type_labels_validation = type_labels[validation_indices]
    type_labels_test = type_labels[test_indices]
    
    # Declare batch size and placeholders
        
    localtime = time.asctime( time.localtime(time.time()) )
    print("Setup Complete :", localtime)

#*****************************************************************************
    
def print_trainable_variables(train_variables):
    for v in train_variables:
        print(v.name)
        
def run_network():
    global saver,quantization
    global speed_labels_test,type_labels_test
    global test_pred_speeds,test_pred_types
    global train_pred_speeds,train_pred_types
    
    # Placeholders
    
    x_data = tf.placeholder(shape=(None, sample_length,3), dtype=precision)
    y_target_speeds = tf.placeholder(shape=(None), dtype=tf.int32)
    y_target_types = tf.placeholder(shape=(None), dtype=tf.int32)
    
    
    # Change functions from here on out if architecture changes
    
    kernel_init = tf.glorot_uniform_initializer(
                                                seed=None,
                                                dtype=precision)

#------------------------------------------------------------------------------
        
# Set up Computation Graph 
    
# COMMON LAYERS
    
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
    
#------------------------------------------------------------------------------
    
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

#------------------------------------------------------------------------------
    
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

#------------------------------------------------------------------------------
        
# SPEEDS LAYERS

# Fully Connected Layers
    
    
    fc_f_speeds = tf.layers.dense(
                            conv1d_flat,
                            hidden_layer_nodes_speeds, 
                            activation=tf.nn.relu,
                            name = "logits_speed"
                            )
    fc_f_16_speeds = tf.cast(fc_f_speeds,precision)  
    fprob_speeds = tf.nn.softmax(fc_f_16_speeds)

#------------------------------------------------------------------------------
    
# TYPES LAYERS

# Fully Connected Layers
    
    
    fc_f_types = tf.layers.dense(
                            conv1d_flat,
                            hidden_layer_nodes_types, 
                            activation=tf.nn.relu,
                            name = "logits_types"
                            )
    fc_f_16_types = tf.cast(fc_f_types,precision)  
    fprob_types = tf.nn.softmax(fc_f_16_types)

#------------------------------------------------------------------------------
    
    localtime = time.asctime( time.localtime(time.time()) )
    print("Graph Defined :", localtime)
    
#*****************************************************************************
    
    saver = tf.train.Saver()
    
# Define Loss function
    
    loss_speeds = tf.losses.sparse_softmax_cross_entropy(labels=y_target_speeds,logits=fc_f_16_speeds)
    
    loss_types = tf.losses.sparse_softmax_cross_entropy(labels=y_target_types,
                                                  logits=fc_f_16_types)
    
    loss = loss_speeds + loss_types
    
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
    order = [conv1d_1,conv1d_2, conv1d_f,fc_f_16_speeds]
    
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
                rand_y_speeds = speed_labels_train[rand_index]
                rand_y_types = type_labels_train[rand_index]
                
                # Training step
                _, temp_loss,get_pred_speeds,get_pred_types = sess.run([train_step,loss,fprob_speeds,fprob_types], 
                                                 feed_dict={x_data : np.array([rand_x]).reshape(batch_size,sample_length,input_channels), 
                                                            y_target_speeds:rand_y_speeds,
                                                            y_target_types:rand_y_types} )
                
            loss_vec.append(temp_loss)
            
            #After the epoch is done, calculate loss, training, validation, test accuracy
            #At the end the following are plotted loss_vec, test_loss, success_rate, test_rate
            
            #Training related loss and prediction
            
            temp_loss_t,train_pred_speeds,train_pred_types = sess.run([loss,fprob_speeds,fprob_types], 
                                            feed_dict={x_data : np.array([x_vals_train]).reshape((7200,sample_length,input_channels)), 
                                                       y_target_speeds:speed_labels_train,
                                                       y_target_types:type_labels_train} )
            loss_vec.append(temp_loss_t)
            
            guess_speeds            = np.argmax(train_pred_speeds,axis=1)
            correct_pred_speeds     = np.sum(np.equal(guess_speeds,speed_labels_train))
            train_accuracy_speeds   = round(correct_pred_speeds*100/len(speed_labels_train),4)
            
            guess_types             = np.argmax(train_pred_types,axis=1)
            correct_pred_types      = np.sum(np.equal(guess_types,type_labels_train))
            train_accuracy_types    = round(correct_pred_types*100/len(type_labels_train),4)
            
            # Get testing loss
            
            test_temp_loss,test_pred_speeds,test_pred_types = sess.run([loss,fprob_speeds,fprob_types], 
                                              feed_dict={x_data : np.array([x_vals_test]).reshape((900,sample_length,input_channels)), 
                                                       y_target_speeds:speed_labels_test,
                                                       y_target_types:type_labels_test} )
            test_loss.append(test_temp_loss)
            
            test_guess_speeds = np.argmax(test_pred_speeds,axis=1)
            test_correct_pred_speeds = np.sum(np.equal(test_guess_speeds,speed_labels_test))
            test_accuracy_speeds = round(test_correct_pred_speeds*100/len(speed_labels_test),4)
            
            test_guess_types = np.argmax(test_pred_types,axis=1)
            test_correct_pred_types = np.sum(np.equal(test_guess_types,type_labels_test))
            test_accuracy_types = round(test_correct_pred_types*100/len(type_labels_test),4)
            
            # Print updates
            
            print('Generation: ' + str("{0:0=5d}".format(i+1)))
            print('SPEEDS')
            print('Training Acc = ' + str((train_accuracy_speeds))+'. Test Acc = ' + str((test_accuracy_speeds)))
            print('TYPES')
            print('Training Acc = ' + str((train_accuracy_types))+'. Test Acc = ' + str((test_accuracy_types)))
            print('Loss = '  + str(round(temp_loss_t,4)))
            
            print()
                
            if (i+1)%50==0:
                
                qweights = []
                weights = []
                
                
                with tf.variable_scope("foo",reuse=tf.AUTO_REUSE):
                    weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(conv1d_1.name)[0] + '/kernel:0')))
                    weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(conv1d_2.name)[0] + '/kernel:0')))
                    weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(conv1d_f.name)[0] + '/kernel:0')))
                    weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_f_speeds.name)[0] + '/kernel:0')))
                    weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_f_types.name)[0] + '/kernel:0')))
                    
                    conv_f_weights = sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(conv1d_f.name)[0] + '/kernel:0'))
                    conv1d_f_over_time.append(np.reshape(np.array(conv_f_weights),(-1)))
                    #plot_histogram(conv_f_weights)
                    
                    
                if(quantization):
                    print("Quantizing...")
                    for h in range(len(order)):
                        qweights.append(quantize_tensor(tf.get_default_graph().get_tensor_by_name(os.path.split(order[h].name)[0] + '/kernel:0'),8))
                        quantize_tensor(tf.get_default_graph().get_tensor_by_name(os.path.split(order[h].name)[0] + '/bias:0'),8)
    
            val_temp_loss,val_pred_speeds,val_pred_types = sess.run([loss,fprob_speeds,fprob_types], 
                                              feed_dict={x_data : np.array([x_vals_validation]).reshape((900,sample_length,input_channels)), 
                                                       y_target_speeds:speed_labels_validation,
                                                       y_target_types:type_labels_validation} )
    
            validation_guess_speeds = np.argmax(val_pred_speeds,axis=1)
            validation_correct_pred_speeds = np.sum(np.equal(validation_guess_speeds,speed_labels_validation))
            validation_accuracy_speeds = round(validation_correct_pred_speeds*100/len(speed_labels_validation),4)
            
            validation_guess_types = np.argmax(val_pred_types,axis=1)
            validation_correct_pred_types = np.sum(np.equal(validation_guess_types,type_labels_validation))
            validation_accuracy_types = round(validation_correct_pred_types*100/len(type_labels_validation),4)
            
            # Plot values
            
            localtime = time.asctime( time.localtime(time.time()) )
        print('SPEEDS')
        print("Validation Accuracy = "+str(validation_accuracy_speeds))
        print('TYPES')
        print("Validation Accuracy = "+str(validation_accuracy_types))
        print()
                      
        #Training related loss and prediction
        
        temp_loss_t,get_pred_speeds,get_pred_types = sess.run([loss,fprob_speeds,fprob_types], 
                                            feed_dict={x_data : np.array([x_vals_train]).reshape((7200,sample_length,input_channels)), 
                                                       y_target_speeds:speed_labels_train,
                                                       y_target_types:type_labels_train} )
        loss_vec.append(temp_loss_t)
        
        guess_speeds            = np.argmax(get_pred_speeds,axis=1)
        correct_pred_speeds     = np.sum(np.equal(guess_speeds,speed_labels_train))
        train_accuracy_speeds   = round(correct_pred_speeds*100/len(speed_labels_train),4)
        
        guess_types             = np.argmax(get_pred_types,axis=1)
        correct_pred_types      = np.sum(np.equal(guess_types,type_labels_train))
        train_accuracy_types    = round(correct_pred_types*100/len(type_labels_train),4)
        
        # Get testing loss
        
        test_temp_loss,test_pred_speeds,test_pred_types = sess.run([loss,fprob_speeds,fprob_types], 
                                          feed_dict={x_data : np.array([x_vals_test]).reshape((900,sample_length,input_channels)), 
                                                   y_target_speeds:speed_labels_test,
                                                   y_target_types:type_labels_test} )
        test_loss.append(test_temp_loss)
        
        test_guess_speeds = np.argmax(test_pred_speeds,axis=1)
        test_correct_pred_speeds = np.sum(np.equal(test_guess_speeds,speed_labels_test))
        test_accuracy_speeds = round(test_correct_pred_speeds*100/len(speed_labels_test),4)
        
        test_guess_types = np.argmax(test_pred_types,axis=1)
        test_correct_pred_types = np.sum(np.equal(test_guess_types,type_labels_test))
        test_accuracy_types = round(test_correct_pred_types*100/len(type_labels_test),4) 
        
        # Get Validation Loss
    
        val_temp_loss,val_pred_speeds,val_pred_types = sess.run([loss,fprob_speeds,fprob_types], 
                                              feed_dict={x_data : np.array([x_vals_validation]).reshape((900,sample_length,input_channels)), 
                                                       y_target_speeds:speed_labels_validation,
                                                       y_target_types:type_labels_validation} )
    
        validation_guess_speeds = np.argmax(val_pred_speeds,axis=1)
        validation_correct_pred_speeds = np.sum(np.equal(validation_guess_speeds,speed_labels_validation))
        validation_accuracy_speeds = round(validation_correct_pred_speeds*100/len(speed_labels_validation),4)
        
        validation_guess_types = np.argmax(val_pred_types,axis=1)
        validation_correct_pred_types = np.sum(np.equal(validation_guess_types,type_labels_validation))
        validation_accuracy_types = round(validation_correct_pred_types*100/len(type_labels_validation),4)
        
        print("QUANTIZED")
        print('SPEEDS')
        print('Training Acc = ' + str((train_accuracy_speeds))+'. Test Acc = ' + str((test_accuracy_speeds)))
        print("Validation Accuracy = "+str(validation_accuracy_speeds))
        print('TYPES')
        print('Training Acc = ' + str((train_accuracy_types))+'. Test Acc = ' + str((test_accuracy_types)))
        print("Validation Accuracy = "+str(validation_accuracy_types))
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
    setup()
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
    