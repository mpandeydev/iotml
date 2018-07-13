import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import csv

hidden_layer_nodes = 1000
hidden_layer_nodes_2 = 700
hidden_layer_nodes_3 = 300
hidden_layer_nodes_4 = 5
hidden_layer_nodes_5 = 500
hidden_layer_nodes_6 = 200
hidden_layer_nodes_7 = 100

hidden_layer_nodes_f = 5 # Must be the same as output logits

step_size           = 0.05
 
samplen             = 9000
batch_size          = 200
 
ffgenerations         = 15000

sample_length       = 4800

filter_width        = 8
kernel_stride       = 10
filter_count        = 6

pool_sizes          = 4
pool_stride         = 4

conv_size           = 570

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)

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


def setup():
    global x_vals_train, x_vals_test, x_vals_validation, y_vals_test, y_vals_train, y_vals_validation
    localtime = time.asctime( time.localtime(time.time()) )
    print("Start time :", localtime)
    
    # Architecture of Network --> get weights here
    
    # Load data from csv
    mag_dataset = []
    choice = []
    
    with open('../training_data.csv', 'r') as f:  #opens PW file
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
    
    y_vals_train = np.array(y_vals_train)
    y_vals_validation = np.array(y_vals_validation)
    y_vals_test = np.array(y_vals_test)

    # Declare batch size and placeholders
        
    localtime = time.asctime( time.localtime(time.time()) )
    print("Setup Complete :", localtime)

#*****************************************************************************
    
def run_network(fc , fc2):
    pool_sizes = fc
    conv_size = fc2
    print("Parameter is : ",fc )
     # Placeholders
    x_data = tf.placeholder(shape=(None, sample_length,1), dtype=tf.float32)
    y_target = tf.placeholder(shape=(None), dtype=tf.int32)
    
    
    # Change functions from here on out if architecture changes
    
    #filterz = tf.random_normal(shape=[3], dtype=tf.float32)
    #filterz = tf.expand_dims(filterz, axis = 2)
    #conv2 = tf.nn.conv1d(x_data,filterz, stride=2, padding="VALID")
    filter_1d = np.array(np.random.rand(filter_width))
    filter_1d = tf.convert_to_tensor(filter_1d,dtype=np.float32)
    filter_1d = tf.reshape(filter_1d,[filter_width,1,1])
    
    #filter_ip = x_data#tf.reshape(x_data,shape=[tf.shape(x_data)[0]/4800,4800,1])
    #dense = tf.layers.dense(inputs=conv2, units=1024, activation=tf.nn.relu)
     
    # Set up Computation Graph 
    #conv1d = tf.squeeze(tf.nn.conv1d(value=x_data,filters=filter_1d,stride=kernel_stride,padding="VALID"))
    conv1d = tf.layers.conv1d(inputs=x_data,filters=filter_count,kernel_size=filter_width,strides=kernel_stride,padding="valid",activation=tf.nn.relu)
    #conv1d_flat = tf.reshape(conv1d, [-1, int(sample_length/kernel_stride)-1])
    pool1d = tf.layers.max_pooling1d(inputs=conv1d, pool_size=pool_sizes, strides=pool_stride)
    #conv1d_flat = tf.reshape(pool1d, [-1, int(sample_length*filter_count/pool_sizes) ])
    conv1d_flat = tf.reshape(pool1d, [-1, conv_size])
    fc_1 = tf.layers.dense(conv1d_flat,hidden_layer_nodes,activation=tf.nn.relu)
    fc_2 = tf.layers.dense(fc_1,hidden_layer_nodes_2,activation=tf.nn.relu)
    fc_3 = tf.layers.dense(fc_2,hidden_layer_nodes_3,activation=tf.nn.relu)
    fc_f = tf.layers.dense(fc_3,hidden_layer_nodes_4,activation=tf.nn.relu)
    '''fc_5 = tf.layers.dense(fc_4,hidden_layer_nodes_5,activation=tf.nn.relu)
    fc_6 = tf.layers.dense(fc_5,hidden_layer_nodes_6,activation=tf.nn.relu)
    fc_7 = tf.layers.dense(fc_6,hidden_layer_nodes_7,activation=tf.nn.relu)
    fc_f = tf.layers.dense(fc_7,hidden_layer_nodes_f,activation=tf.nn.relu)'''
    fprob = tf.nn.softmax(fc_f, name=None)
    
    localtime = time.asctime( time.localtime(time.time()) )
    print("Graph Defined :", localtime)
    
    #*****************************************************************************
    
    # Define Loss function
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_target,logits=fc_f)
    #loss = tf.losses.softmax_cross_entropy(onehot_labels=y_target,logits=final_out)
    
    # Optimizer function
    
    my_opt = tf.train.AdagradOptimizer(step_size)
    #my_opt = tf.train.AdagradOptimizer(step_size)
    train_step = my_opt.minimize(loss)
    
    # Initialize all variables
    
    init = tf.global_variables_initializer()
    sess.run(init)
    localtime = time.asctime( time.localtime(time.time()) )
    print("Initialized Variables:", localtime)
    
    # Log vectors
    
    loss_vec = []
    test_loss = []
    test_pred = []
    pred_vec = []
    success_rate = []
    successful_guesses = []
    test_rate = []
    # Train 
    
    for i in range(generations):
        with tf.Graph().as_default():
        
            # Get random batch
            
            rand_index = np.random.choice(len(x_vals_train), size = batch_size)
            rand_x = [x_vals_train[rand_index]]
            rand_y = y_vals_train[rand_index]
            
            # Training step
            
            sess.run(train_step, feed_dict={x_data : np.array([rand_x]).reshape((batch_size,4800,1)), y_target:rand_y})
            temp_loss,get_pred = sess.run([loss,fprob], feed_dict={x_data : np.array([rand_x]).reshape((batch_size,4800,1)), y_target:rand_y} )
            loss_vec.append(temp_loss)
            
            temp_loss,get_pred = sess.run([loss,fprob], feed_dict={x_data : np.array([x_vals_train]).reshape((7200,4800,1)), y_target:y_vals_train} )
            guess = np.argmax(get_pred,axis=1)
            pred_vec.append(guess)
            correct_pred = np.sum(np.equal(guess,y_vals_train))
            successful_guesses.append(correct_pred)
            success_rate.append(correct_pred/len(x_vals_train) )
            train_accuracy = round(correct_pred*100/len(x_vals_train),4)
            
            # Get testing loss
            test_temp_loss,predict = sess.run([loss,fprob], feed_dict={x_data : np.array([x_vals_test]).reshape((900,4800,1)), y_target:y_vals_test})
            test_loss.append(test_temp_loss)
            test_pred.append(predict)
            test_guess = np.argmax(predict,axis=1)
            test_correct_pred = np.sum(np.equal(test_guess,y_vals_test))
            test_rate.append(test_correct_pred/len(y_vals_test))
            test_accuracy = round(test_correct_pred*100/len(y_vals_test),4)
            
            # Print updates
            if (i+1)%100==0:
                #localtime = time.asctime( time.localtime(time.time()) )
                #time_now = datetime.datetime.now().time()
                print('Generation: ' + str("{0:0=5d}".format(i+1)) + '. Training Acc = ' + str((train_accuracy))+'. Test Acc = ' + str((test_accuracy))+ '. Loss = '  + str(round(temp_loss,4)))
        
    validation_loss,validation_predict = sess.run([loss,fprob], feed_dict={x_data : np.array([x_vals_validation]).reshape((900,4800,1)), y_target:y_vals_validation})
    validation_guess = np.argmax(validation_predict,axis=1)
    validation_correct_pred = np.sum(np.equal(validation_guess,y_vals_validation))
    validation_accuracy = round(validation_correct_pred*100/len(y_vals_validation),4)
    
    print("Validation Accuracy = "+str(validation_accuracy))
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
    
    plt.plot(success_rate, 'r--', label='Training Success rate')
    plt.plot(test_rate, 'b--', label='Test Success rate')
    plt.title('Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Success Rate')
    plt.legend(loc='upper right')
    plt.show()
    
#setup()
#run_network(2)

#setup()
#run_network(2)

setup()
run_network(4, 63789429)
