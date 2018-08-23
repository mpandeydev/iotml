import tensorflow as tf
import tfgraphviz as tfg
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import csv

import os

#tf.enable_eager_execution()

os.environ["PATH"] += os.pathsep + 'D:/CMU_Summer18/graphviz-2.38/release/bin/'

precision = tf.float32
logit_size = 8

hidden_layer_nodes = 900
hidden_layer_nodes_2 = 500
#hidden_layer_nodes_3 = 100

hidden_layer_nodes_f = logit_size # Must be the same as output logits

step_size           = 0.02
 
samplen             = 9000
batch_size          = 200
input_channels      = 3
 
generations         = 9000
loss_limit          = 0.075

sample_length       = 1600

filter_width        = 8
kernel_stride       = 10
feature_maps        = 5

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
    
def quantize_tensor(input_tensor,bits_q):

    #if(quantize_on == False):
        #return input_tensor
    
    bits = bits_q
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

def op_convert_to_int8(i):
    quantized_tensor = []
    zero = tf.cast(0,tf.float32)
    one = tf.cast(1,tf.float32)
    # Round weights to nearest part
    
    def neg(): 
        quantized_tensor.append(0)
        return 0
    def pos(): 
        quantized_tensor.append(1)
        return 1
    r = tf.cond(tf.greater(zero, i[0]), neg, pos)
    return tf.cast(quantized_tensor,tf.float32)
        
def convert_tensor_to_int8(input_tensor):
    full_range = 256
    input_array = input_tensor
    
    max_value = tf.reduce_max(input_array)
    min_value = tf.reduce_min(input_array)
    
    flat_x_vals = tf.layers.flatten(input_array)
    quantized_tensor = []
    
    quantized_tensor = tf.map_fn(op_convert_to_int8,flat_x_vals)
    
    #converted_array = sess.run(quantized_tensor)
    #print(converted_array)
    converted_array_t = tf.cast(quantized_tensor,dtype=np.float16)
    #converted_array_t = tf.reshape(converted_array_t, tf.shape(input_tensor))
    #assign_op = tf.assign(input_tensor,converted_array_t)
    #sess.run(assign_op)
    return converted_array_t   


#--------------------------------------------------------------------------------------------------------------

def import_npy():
    global x_vals,speeds,types
    tdata = np.load('../../training_data_3d.npy')
    print(np.size(tdata))
    speeds = tdata[:,3,0]
    types = tdata[:,3,1]
    x_vals = x_vals = tdata[:,0:3,:]
    original_shape = x_vals.shape
    x_vals = x_vals.reshape(-1,1600)
    means = np.mean(x_vals,dtype=np.float16,axis=(1))
    means = means.reshape(27024,1)
    x_vals = x_vals-means
    x_vals = x_vals.reshape(original_shape)
    
    #x_vals = np.load('../../q_input.npy')
    #x_vals = np.load('../dataset_quantized.npy')
    
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
def print_trainable_variables(train_variables):
    for v in train_variables:
        print(v.name)
        
def run_network():
    global saver
    conv_size = 160*feature_maps
    
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

    #conv1d_f = tf.layers.conv1d(inputs=x_data,filters=feature_maps,kernel_size=filter_width,strides=kernel_stride,padding="valid",activation=tf.nn.relu,name="conv1d_f")
    conv1d_f = tf.layers.conv1d(inputs=x_data,filters=feature_maps,kernel_size=filter_width,strides=kernel_stride,padding="valid",activation=tf.nn.relu)
    conv1d_flat = tf.reshape(conv1d_f, [-1, conv_size])#How is conv_size known?
    conv1d_flat_int8 = convert_tensor_to_int8(conv1d_flat)
    # TO DO : Convert to Int8 Here
    
    # Fully Connected Layers
    
    fc_1 = tf.layers.dense(conv1d_flat_int8,hidden_layer_nodes,activation=tf.nn.relu)
    fc_2 = tf.layers.dense(fc_1,hidden_layer_nodes_2,activation=tf.nn.relu)
    #fc_3 = tf.layers.dense(fc_2,hidden_layer_nodes_3,activation=tf.nn.relu)
    fc_f = tf.layers.dense(fc_2,hidden_layer_nodes_f,activation=tf.nn.relu)
    fprob = tf.nn.softmax(fc_f)
    
    localtime = time.asctime( time.localtime(time.time()) )
    print("Graph Defined :", localtime)
    
    #g = tfg.board(tf.get_default_graph())
    #g.view()
    #*****************************************************************************
    
    saver = tf.train.Saver()
    
# Define Loss function
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_target,logits=fc_f)#Why fc_f and not fprob?
    #loss = tf.losses.softmax_cross_entropy(onehot_labels=y_target,logits=final_out)
    
# Optimizer function
    
    my_opt = tf.train.AdagradOptimizer(step_size)
    
    # First quantization cycle
    train_vars = tf.trainable_variables() 
    train_step = my_opt.minimize(loss,var_list=train_vars)
    
    '''# Second quantization cycle
    train_vars.remove(tf.get_variable('conv1d_f/bias',[feature_maps]))
    train_vars.remove(tf.get_variable('conv1d_f/kernel',[filter_width,3,feature_maps]))
    train_step_5 = my_opt.minimize(loss,var_list = train_vars)
    
    # Fifth quantization cycle
    train_vars.remove(tf.get_variable('dense/bias',[hidden_layer_nodes]))
    train_vars.remove(tf.get_variable('dense/kernel',[conv_size,hidden_layer_nodes]))
    train_step_4 = my_opt.minimize(loss,var_list = train_vars)
    
    # Fourth quantization cycle
    train_vars.remove(tf.get_variable('dense_1/bias',[hidden_layer_nodes_2]))
    train_vars.remove(tf.get_variable('dense_1/kernel',[hidden_layer_nodes,hidden_layer_nodes_2]))
    train_step_3 = my_opt.minimize(loss,var_list = train_vars)
    
    # Third quantization cycle
    #train_vars.remove(tf.get_variable('dense_2/bias',[hidden_layer_nodes_3]))
    #train_vars.remove(tf.get_variable('dense_2/kernel',[hidden_layer_nodes_2,hidden_layer_nodes_3]))
    #train_step_2 = my_opt.minimize(loss,var_list = train_vars)
    
    # Second quantization cycle
    #train_vars.remove(tf.get_variable('dense_3/bias',[logit_size]))
    #train_vars.remove(tf.get_variable('dense_3/kernel',[hidden_layer_nodes_3,logit_size]))
    #train_step_1 = my_opt.minimize(loss,var_list = train_vars)'''
    
# Initialize all variables
    
    init = tf.global_variables_initializer()
    sess.run(init)
    localtime = time.asctime( time.localtime(time.time()) )
    print("Initialized Variables:", localtime)
    
    # Log vectors
    
    loss_vec = []
    test_loss = []
    test_pred = []
    #pred_vec = []
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
    
    order = [conv1d_f,fc_1,fc_2,fc_f]
    #optimizer_order = [train_step,train_step_5,train_step_4,train_step_3]
    
    for h in range(quantize_train):
        print( )
        print("Quantize Train Loop iteration : ",h)
        print( )
        print_trainable_variables(train_vars)
        print( )
        i = 0
        for i in range(generations):
        #while(temp_loss_t>loss_limit):
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
                #_, temp_loss,get_pred = sess.run([optimizer_order[h],loss,fprob], feed_dict={x_data : np.array([rand_x]).reshape((batch_size,sample_length,input_channels)), y_target:rand_y} )
                _, temp_loss,get_pred, int8_values = sess.run([train_step,loss,fprob,conv1d_flat_int8], feed_dict={x_data : np.array([rand_x]).reshape(batch_size,sample_length,input_channels), y_target:rand_y} )
            
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
            if (i+1)%20==0:
            #if True:
                #t_weights = sess.run(tf.get_default_graph().get_tensor_by_name(os.path.split(fc_2.name)[0] + '/kernel:0'))
                #print("Pre Quantize Unique Weights : ", len(np.unique(t_weights)))
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
        
        qweights.append(quantize_tensor(tf.get_default_graph().get_tensor_by_name(os.path.split(order[h].name)[0] + '/kernel:0'),8))
        qweights.append(quantize_tensor(tf.get_default_graph().get_tensor_by_name(os.path.split(order[h].name)[0] + '/bias:0'),8))
                
        
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
        
    
    #saver.save(sess, "../quantizeded_model_speeds_int8.ckpt")
    
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
    
    return weights,qweights,int8_values

#______________________________________________________________________________

def save_network():
    return saver.save(sess, "../quantizeded_model_speeds_int8.ckpt")

with tf.variable_scope("foo",reuse=tf.AUTO_REUSE):
    import_npy()
    setup("speeds")
    weights,weights_q,int8_values = run_network()
    variables_names = [v.name for v in tf.trainable_variables()]
    for i in weights_q:
        print(len(np.unique(i)))
    #sess.close()