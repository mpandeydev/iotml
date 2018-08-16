import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import time

precision = tf.float32

t_logit_size = 8
s_logit_size = 8

model = "../quantizeded_model_types.ckpt"

hidden_layer_nodes = 900

#hidden_layer_nodes_s = 1000
hidden_layer_nodes_2_s = 500
hidden_layer_nodes_3_s = s_logit_size

#   hidden_layer_nodes_t = 1000
# hidden_layer_nodes_2_t = 500
# hidden_layer_nodes_3_t = t_logit_size

step_size           = 0.05
 
samplen             = 9005
batch_size          = 500
input_channels      = 3
 
generations         = 100

sample_length       = 1600

filter_width        = 8
kernel_stride       = 10
feature_maps       = 5

pool_sizes          = 4
pool_stride         = 4

conv_size           = 800

thisgraph = tf.Graph()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

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
    global x_vals, y_vals, speeds, types, x_vals_train, x_vals_test, x_vals_validation, y_vals_test_t, y_vals_train_t, y_vals_validation_t,y_vals_test_s, y_vals_train_s, y_vals_validation_s,logit_size,saver
    localtime = time.asctime( time.localtime(time.time()) )
    print("Start time :", localtime)  
    
    x_vals = np.array([x[0:sample_length] for x in x_vals])
    choice = np.random.choice(len(x_vals), size = samplen)
    x_vals = x_vals[np.array(choice)]
    
    x_vals = x_vals.astype(np.float32)
    x_vals = tf.keras.utils.normalize(x_vals,axis=-1,order=2)
    
    train_indices = np.random.choice(len(x_vals), round(len(x_vals)*1), replace=False)
 
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
    y_vals_train_s = y_vals_s[train_indices]    
    y_vals_train_t = y_vals_t[train_indices]
    
    # Declare batch size and placeholders
        
    localtime = time.asctime( time.localtime(time.time()) )
    print("Setup Complete :", localtime)

#*****************************************************************************
import_npy()
setup()

 # Placeholders
x_data = tf.placeholder(shape=(None, sample_length,3), dtype=tf.float32)
y_target_s = tf.placeholder(shape=(None), dtype=tf.int32) 
y_target_t = tf.placeholder(shape=(None), dtype=tf.int32) 

# Change functions from here on out if architecture changes
     
# Set up Computation Graph 

# Convolutional and Pooling Layers
conv1d_f = tf.layers.conv1d(inputs=x_data,filters=feature_maps,kernel_size=filter_width,strides=kernel_stride,padding="valid",activation=tf.nn.relu,name="foo/conv1d_f")
#conv1d_f = tf.layers.max_pooling1d(inputs=conv1d_f, pool_size=pool_sizes, strides=pool_stride)
conv1d_flat = tf.reshape(conv1d_f, [-1, conv_size],)
tf.cast(conv1d_flat,precision)
#cmin = tf.reduce_max(conv1d_flat)
# Fully Connected Layers

fc_1 = tf.layers.dense(conv1d_flat,hidden_layer_nodes,activation=tf.nn.relu,name="foo/dense")
tf.cast(fc_1,precision)

#fc_1_s = tf.layers.dense(conv1d_flat,hidden_layer_nodes_s,activation=tf.nn.relu)

fc_2_s = tf.layers.dense(fc_1,hidden_layer_nodes_2_s,activation=tf.nn.relu,name="foo/dense_1")
tf.cast(fc_2_s,precision)
fc_f_s = tf.layers.dense(fc_2_s,hidden_layer_nodes_3_s,activation=tf.nn.relu,name="foo/dense_2")
tf.cast(fc_2_s,precision)
fprob_s = tf.nn.softmax(fc_f_s, name="foo/softmax")

#fc_1_t = tf.layers.dense(conv1d_flat,hidden_layer_nodes_t,activation=tf.nn.relu)

#fc_2_t = tf.layers.dense(fc_1  ,hidden_layer_nodes_2_t,activation=tf.nn.relu,name="fc_2_t")
#tf.cast(fc_2_t,precision)
#fc_f_t = tf.layers.dense(fc_2_t,hidden_layer_nodes_3_t,activation=tf.nn.relu,name="fc_f_t")
#tf.cast(fc_f_t,precision)
#fprob_t = tf.nn.softmax(fc_f_t, name="softmax_t")

loss = tf.losses.sparse_softmax_cross_entropy(labels=y_target_s,logits=fc_f_s)
#types_loss = tf.losses.sparse_softmax_cross_entropy(labels=y_target_t,logits=fc_f_t)
#loss = tf.add(speeds_loss,types_loss)
qsaver = tf.train.Saver()
#tf.contrib.quantize.create_eval_graph(thisgraph)
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

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
# Restore variables from disk.
saver.restore(sess, model)
print("Model restored.")

conv1d_w = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'foo/conv1d_f/kernel'))[0]
fc_1 = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'foo/dense/kernel'))[0]
fc_2 = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'foo/dense_1/kernel'))[0]
fc_f = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'foo/dense_2/kernel'))[0]
fprob = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'softmax_layer'))
#fc_2_t_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc_2_t')
#fc_f_t_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc_f_t')
#fprob_t_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'softmax_t')
#all_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

#max_range, min_range = quantize_layer(conv1d_w)

for i in range(generations):
     with thisgraph.as_default():
            
         # Get random batch
                
            rand_index = np.random.choice(len(x_vals_train), size = batch_size)
            rand_x = [x_vals_train[rand_index]]
                
            rand_y_s = y_vals_train_t[rand_index]
            #rand_y_t = y_vals_train_t[rand_index]
            
            temp_loss,pred_training_s = sess.run([loss,fprob_s], feed_dict={x_data : np.array([rand_x]).reshape((batch_size,sample_length,input_channels)), y_target_s:rand_y_s} )
            loss_vec.append(temp_loss)
            
            #temp_loss,pred_training_s,pred_training_t = sess.run([loss,fprob_s,fprob_t], feed_dict={x_data : np.array([x_vals_train]).reshape((9005,sample_length,input_channels)), y_target_s:y_vals_train_s, y_target_t:y_vals_train_t} )
            
            guess_s = np.argmax(pred_training_s,axis=1)
            #guess_t = np.argmax(pred_training_t,axis=1)
            
            pred_vec_s.append(guess_s)
            #pred_vec_t.append(guess_t)
            
            correct_pred_s = np.sum(np.equal(guess_s,rand_y_s))
            #correct_pred_t = np.sum(np.equal(guess_t,rand_y_t))
            
            
            successful_guesses_s.append(correct_pred_s)
            #successful_guesses_t.append(correct_pred_t)
            
            #success_rate_t.append(correct_pred_t/len(rand_y_t) )
            #train_accuracy_t = round(correct_pred_t*100/len(rand_y_t),4)
            
            success_rate_s.append(correct_pred_s/len(rand_y_s) )
            train_accuracy_s = round(correct_pred_s*100/len(rand_y_s),4)
            
            
            print('Generation: ' + str("{0:0=5d}".format(i+1))) 
            print('SPEEDS : Inference Acc = ' + str((train_accuracy_s))) 
            #print('TYPES : Inference Acc = ' + str((train_accuracy_t))) 
            print('Loss = '  + str(round(temp_loss,4)))
            print()

accuracy_mean_s = np.mean(success_rate_s)
accuracy_stdev_s = np.std(success_rate_s)
print("SPEEDS : Mean = ",round(accuracy_mean_s,4)," Std Dev = ",round(accuracy_stdev_s,4))
#accuracy_mean_t = np.mean(success_rate_t)
#accuracy_stdev_t = np.std(success_rate_t)
#print("TYPES : Mean = ",round(accuracy_mean_t,4)," Std Dev = ",round(accuracy_stdev_t,4))

# Plot values
'''plt.hist(success_rate_s,bins=10)
plt.title('Speeds Accuracies')
plt.show()

plt.hist(success_rate_t,bins=10)
plt.title('Types Accuracies')
plt.show()'''

sess.close()
            
            