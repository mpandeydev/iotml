import tensorflow as tf
import tfgraphviz as tfg
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import csv

import os
os.environ["PATH"] += os.pathsep + 'D:/CMU_Summer18/graphviz-2.38/release/bin/'

precision = tf.float32
logit_size = 8

hidden_layer_nodes = 900
hidden_layer_nodes_2 = 450
hidden_layer_nodes_3 = 100

hidden_layer_nodes_f = logit_size # Must be the same as output logits

step_size           = 0.015
 
samplen             = 9000
batch_size          = 200
input_channels      = 3
 
generations         = 10
loss_limit          = 0.2

sample_length       = 1600

filter_width        = 8
kernel_stride       = 10
feature_maps        = 6

pool_sizes          = 4
pool_stride         = 4

conv_size           = 104
quantize_train      = 1

quantize_on     = True

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

# Quantization Function
    
def quantize_tensor(input_tensor):

    #if(quantize_on == False):
        #return input_tensor
    
    bits = 8
    full_range = 2**bits
    converted_array = []
    
    input_array = sess.run(input_tensor)
    max_value = np.float16(np.amax(input_array)) 
    min_value = np.float16(np.amin(input_array))
    quantum = np.float16((max_value-min_value)/full_range)
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

#--------------------------------------------------------------------------------------------------------------

def import_npy():
    global x_vals,speeds,types
    tdata = np.load('../training_data_3d.npy')
    speeds = tdata[:,3,0]
    types = tdata[:,3,1]
    #x_vals = x_vals = tdata[:,0:3,:]
    x_vals = np.load('../dataset_quantized.npy')
    
def setup(dataset):
    global x_vals, y_vals, speeds, types, x_vals_train, x_vals_test, x_vals_validation, y_vals_test, y_vals_train, y_vals_validation,logit_size
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
    
    # Change for datasets
    if(dataset=="speeds"):
        logit_size = 5
        y_vals = np.array([speeds[:]])
    if(dataset=="types"):
        logit_size = 8
        y_vals = np.array([types[:]])
    y_vals = np.array(y_vals[0,np.array(choice)])
    y_vals = y_vals.astype(np.float)
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
def define_network(fmaps, conv_size):
    feature_maps = fmaps
    conv_size = conv_size
    print("Parameter is : ",feature_maps )
     # Placeholders
    x_data = tf.placeholder(shape=(None, sample_length,3), dtype=tf.float32)
    y_target = tf.placeholder(shape=(None), dtype=tf.int32)
    
    
    # Change functions from here on out if architecture changes

    #conv2 = tf.nn.conv1d(x_data,filterz, stride=2, padding="VALID")
    filter_1d = np.array(np.random.rand(filter_width))
    filter_1d = tf.convert_to_tensor(filter_1d,dtype=np.float16)
    filter_1d = tf.reshape(filter_1d,[filter_width,1,1])
         
# Set up Computation Graph 
    
    # Convolutional and Pooling Layers

    conv1d_f = tf.layers.conv1d(inputs=x_data,filters=feature_maps,kernel_size=filter_width,strides=kernel_stride,padding="valid",activation=tf.nn.relu,name="conv1d_f")
    #conv1d_f = tf.layers.max_pooling1d(inputs=conv1d_f, pool_size=pool_sizes, strides=pool_stride)
    conv1d_flat = tf.reshape(conv1d_f, [-1, conv_size])#How is conv_size known?
    
    # Fully Connected Layers
    
    fc_1 = tf.layers.dense(conv1d_flat,hidden_layer_nodes,activation=tf.nn.relu)
    fc_2 = tf.layers.dense(fc_1,hidden_layer_nodes_2,activation=tf.nn.relu)
    fc_3 = tf.layers.dense(fc_2,hidden_layer_nodes_3,activation=tf.nn.relu)
    fc_f = tf.layers.dense(fc_3,hidden_layer_nodes_f,activation=tf.nn.relu)
    fprob = tf.nn.softmax(fc_f, name=None)
    
    localtime = time.asctime( time.localtime(time.time()) )
    print("Graph Defined :", localtime)
    
    #g = tfg.board(tf.get_default_graph())
    #g.view()
    #*****************************************************************************
    
# Define Loss function
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_target,logits=fc_f)#Why fc_f and not fprob?
    #loss = tf.losses.softmax_cross_entropy(onehot_labels=y_target,logits=final_out)
    
# Optimizer function
    
    my_opt = tf.train.AdagradOptimizer(step_size)
    train_step = my_opt.minimize(loss)
    
# Initialize all variables
    
    init = tf.global_variables_initializer()
    sess.run(init)
    localtime = time.asctime( time.localtime(time.time()) )
    print("Initialized Variables:", localtime)
    
    run_network(loss, train_step ,fprob)
    
def run_network(tsetp , final_layer):
    
    # Log vectors
    
    loss_vec = []
    test_loss = []
    test_pred = []
    pred_vec = []
    success_rate = []
    successful_guesses = []
    test_rate = []
    qweights = []
    weights = []
    
    temp_loss_t = 1
    i = 0
    # Train 
    
    for h in range(quantize_train):
        print("Quantize Train Loop iteration : ",h)
        
        for i in range(generations):
        #while(temp_loss_t>loss_limit/(h+1)):
            i+=1
            if(i % 100 == 0):
                localtime = time.asctime( time.localtime(time.time()) )
                #print("Epoch %d start time:" % i, localtime)
    
            dataset_size = len(x_vals)
    
            num_iters = dataset_size//batch_size
    
            for iter in range(num_iters):
    
                #Get random batch
                #rand_x shape (1, batch_size, 3, 1600)
                #rand_y shape (batch_size,)
                rand_index = np.random.choice(len(x_vals_train), size = batch_size)
                rand_x = [x_vals_train[rand_index]]
                rand_y = y_vals_train[rand_index]
    
                # Training step
                _, temp_loss,get_pred = sess.run([train_step,loss,fprob], feed_dict={x_data : np.array([rand_x]).reshape((batch_size,sample_length,input_channels)), y_target:rand_y} )
            
            loss_vec.append(temp_loss)
            #After the epoch is done, calculate loss, training, validation, test accuracy
            #At the end the following are plotted loss_vec, test_loss, success_rate, test_rate
            
            #Training related loss and prediction
            temp_loss_t,get_pred = sess.run([loss,fprob], feed_dict={x_data : np.array([x_vals_train]).reshape((7200,sample_length,input_channels)), y_target:y_vals_train} )
            loss_vec.append(temp_loss_t)
            guess = np.argmax(get_pred,axis=1)
            correct_pred = np.sum(np.equal(guess,y_vals_train))
            successful_guesses.append(correct_pred)
            success_rate.append(correct_pred/len(x_vals_train) )
            train_accuracy = round(correct_pred*100/len(x_vals_train),4)
            
            # Get testing loss
            test_temp_loss,predict = sess.run([loss,fprob], feed_dict={x_data : np.array([x_vals_test]).reshape((900,sample_length,input_channels)), y_target:y_vals_test})
            test_loss.append(test_temp_loss)
            test_pred.append(predict)
            test_guess = np.argmax(predict,axis=1)
            test_correct_pred = np.sum(np.equal(test_guess,y_vals_test))
            test_rate.append(test_correct_pred/len(y_vals_test))
            test_accuracy = round(test_correct_pred*100/len(y_vals_test),4)
            
            # Print updates
            if (i+1)%500==0:
            #if True:
                
                qweights = []
                weights = []
                
                #t_weights = sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_2.name)[0] + '/kernel:0'))
                #print("Pre Quantize Unique Weights : ", len(np.unique(t_weights)))
                '''
                weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(conv1d_f.name)[0] + '/kernel:0')))
                weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(conv1d_f.name)[0] + '/bias:0')))
                weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_1.name)[0] + '/kernel:0')))
                weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_1.name)[0] + '/bias:0')))
                weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_2.name)[0] + '/kernel:0')))
                weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_2.name)[0] + '/bias:0')))
                weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_f.name)[0] + '/kernel:0')))
                weights.append(sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_f.name)[0] + '/bias:0')))
                 
                
                temp_loss,get_pred = sess.run([loss,fprob], feed_dict={x_data : np.array([x_vals_train]).reshape((7200,sample_length,input_channels)), y_target:y_vals_train} )
                guess = np.argmax(get_pred,axis=1)
                correct_pred = np.sum(np.equal(guess,y_vals_train))
                train_accuracy = round(correct_pred*100/len(x_vals_train),4)
                
                test_temp_loss,predict = sess.run([loss,fprob], feed_dict={x_data : np.array([x_vals_test]).reshape((900,sample_length,input_channels)), y_target:y_vals_test})
                test_guess = np.argmax(predict,axis=1)
                test_correct_pred = np.sum(np.equal(test_guess,y_vals_test))
                test_accuracy = round(test_correct_pred*100/len(y_vals_test),4)
                '''
            
                #print('Generation: ' + str(i+1) + '. Loss = ' + str((temp_loss))+'. Test Loss = ' + str((test_temp_loss))+'. Test Accuracy = ' + str((test_accuracy)))
                #print('Generation: ' + str(i+1) + '. Loss = ' + str((temp_loss))+". Accuracy "+str(correct_pred*100/batch_size)+"%")
                #localtime = time.asctime( time.localtime(time.time()) )
                #time_now = datetime.datetime.now().time()
                print('Generation: ' + str("{0:0=5d}".format(i+1)) + '. Training Acc = ' + str((train_accuracy))+'. Test Acc = ' + str((test_accuracy))+ '. Loss = '  + str(round(temp_loss_t,4)))
    
            validation_loss,validation_predict = sess.run([loss,fprob], feed_dict={x_data : np.array([x_vals_validation]).reshape((900,sample_length,input_channels)), y_target:y_vals_validation})
            validation_guess = np.argmax(validation_predict,axis=1)
            validation_correct_pred = np.sum(np.equal(validation_guess,y_vals_validation))
            validation_accuracy = round(validation_correct_pred*100/len(y_vals_validation),4)
            
            # Plot values
            localtime = time.asctime( time.localtime(time.time()) )
            #print("End time :", localtime)
        print('Generation: ' + str("{0:0=5d}".format(i+1)) + '. Training Acc = ' + str((train_accuracy))+'. Test Acc = ' + str((test_accuracy))+ '. Loss = '  + str(round(temp_loss,4)))
        print("Validation Accuracy = "+str(validation_accuracy))
        print()
        
        qweights.append(quantize_tensor(tf.get_default_graph().get_tensor_by_name(os.path.split(conv1d_f.name)[0] + '/kernel:0')))
        qweights.append(quantize_tensor(tf.get_default_graph().get_tensor_by_name(os.path.split(conv1d_f.name)[0] + '/bias:0')))
        qweights.append(quantize_tensor(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_1.name)[0] + '/kernel:0')))
        qweights.append(quantize_tensor(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_1.name)[0] + '/bias:0')))
        qweights.append(quantize_tensor(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_2.name)[0] + '/kernel:0')))
        qweights.append(quantize_tensor(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_2.name)[0] + '/bias:0')))
        qweights.append(quantize_tensor(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_f.name)[0] + '/kernel:0')))
        qweights.append(quantize_tensor(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_f.name)[0] + '/bias:0')))
        
        #t_weights = sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_2.name)[0] + '/kernel:0'))
        #print("Post Quantize Unique Weights : ", len(np.unique(t_weights)))
        #print( )
                
        #Training related loss and prediction
        temp_loss,get_pred = sess.run([loss,fprob], feed_dict={x_data : np.array([x_vals_train]).reshape((7200,sample_length,input_channels)), y_target:y_vals_train} )
        loss_vec.append(temp_loss)
        guess = np.argmax(get_pred,axis=1)
        correct_pred = np.sum(np.equal(guess,y_vals_train))
        successful_guesses.append(correct_pred)
        success_rate.append(correct_pred/len(x_vals_train) )
        train_accuracy = round(correct_pred*100/len(x_vals_train),4)
        
        # Get testing loss
        test_temp_loss,predict = sess.run([loss,fprob], feed_dict={x_data : np.array([x_vals_test]).reshape((900,sample_length,input_channels)), y_target:y_vals_test})
        test_loss.append(test_temp_loss)
        test_pred.append(predict)
        test_guess = np.argmax(predict,axis=1)
        test_correct_pred = np.sum(np.equal(test_guess,y_vals_test))
        test_rate.append(test_correct_pred/len(y_vals_test))
        test_accuracy = round(test_correct_pred*100/len(y_vals_test),4)    
    
        validation_loss,validation_predict = sess.run([loss,fprob], feed_dict={x_data : np.array([x_vals_validation]).reshape((900,sample_length,input_channels)), y_target:y_vals_validation})
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
    
    
    #sess.close()
    
    return weights,qweights

with tf.variable_scope("foo",reuse=tf.AUTO_REUSE):
    import_npy()
    setup("types")
    weights,weights_q = run_network(15,2400)
    new_var = tf.get_variable('dense_3/bias',[8],trainable=False)
    #last_layer = sess.run(new_var)
    #num_w = np.unique(weights)
    #num_q = np.unique(weights_q)
    variables_names = [v.name for v in tf.trainable_variables()]
    for i in weights:
        print(len(np.unique(i)))
    sess.close()
