import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

test_on = "types" # speeds or types

variable_scope = "foo" #/ "pruned"
model = "../trained_models/pruned_model.ckpt"
#"../trained_models/pure_conv/full_precision_model.ckpt"
#model = "../trained_models/pruned_networks/test_model.ckpt"

logit_size = 8

precision = tf.float32
precision_np = np.float32

hidden_layer_nodes = 900
hidden_layer_nodes_2 = 500

hidden_layer_nodes_f = logit_size # Must be the same as output logits

step_size           = 0.2

samplen             = 9008 
batch_size          = 200
input_channels      = 3
 
generations         = 1000
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

#______________________________________________________________________________

def import_npy():
    global x_vals,speeds,types
    tdata = np.load('../../datasets/training_data_3d.npy')
    speeds = tdata[:,3,0]
    types = tdata[:,3,1]
    x_vals = tdata[:,0:3,:]

def setup(choice):
    global x_vals, y_vals
    global speeds, types 
    global x_vals_train, x_vals_test, x_vals_validation 
    global y_vals_test, y_vals_train, y_vals_validation
    global logit_size,saver,samplen
    
    selected_choice = str(choice)
    localtime = time.asctime( time.localtime(time.time()) )
    print("Start time :", localtime)  
    
    x_vals = np.array([x[0:sample_length] for x in x_vals])
    choice = np.random.choice(len(x_vals), size = samplen)
    x_vals = x_vals[np.array(choice)]
    
    x_vals = x_vals.astype(precision_np)
    
    original_shape = x_vals.shape
    x_vals = x_vals.reshape(-1,1600)
    means = np.mean(x_vals,dtype=precision_np,axis=(1))
    means = means.reshape(27024,1)
    x_vals = x_vals-means
    x_vals = x_vals.reshape(original_shape)
    
    train_indices = np.random.choice(len(x_vals), round(len(x_vals)*1), replace=False)
    
    def labels_to_int(labels):
        out = np.array(labels[0,np.array(choice)])
        out = out.astype(np.float)
        out = out.astype(np.int32)
        return out
    
    if(str(selected_choice)==str("types")):
        print("Dataset Chosen : Types")
        y_vals = labels_to_int(np.array([types[:]]))
        logit_size = 8
        
    if(str(selected_choice)==str("speeds")):
        print("Dataset Chosen : Speeds")
        y_vals = labels_to_int(np.array([speeds[:]]))
        logit_size = 5
        
    else:
        print("You selected : ",selected_choice)
        print("No Dataset Selected.")
    
    localtime = time.asctime( time.localtime(time.time()) )
    print("Loaded Dataset :", localtime)
    
    # Fix seed
    
    seed = 2
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
    # 80:10:10 data split and normalization
    
    x_vals_train = x_vals[train_indices]    
    y_vals_train = y_vals[train_indices]    
    
    # Declare batch size and placeholders
        
    localtime = time.asctime( time.localtime(time.time()) )
    print("Setup Complete :", localtime)

def generate_model(feature_maps,feature_maps_2,feature_maps_3):
 # Placeholders
    x_data = tf.placeholder(shape=(None, sample_length,3), dtype=tf.float32)
    y_target = tf.placeholder(shape=(None), dtype=tf.int32) 
      
    with tf.variable_scope(variable_scope):  
        # First Convolution Layer
        
        conv1d_1 = tf.layers.conv1d(
                                    inputs=x_data,
                                    filters=feature_maps,
                                    kernel_size=filter_width,
                                    strides=kernel_stride,
                                    padding="valid",
                                    activation=tf.nn.relu,
                                    name="conv1d_1"
                                    )
        conv1d_1_cast = tf.cast(conv1d_1,
                                precision_np)
        conv1d_1_norm = tf.layers.batch_normalization(conv1d_1_cast, 
                                                      training = True, 
                                                      fused=False, 
                                                      name = "bn1")
        
        conv1d_1_norm16 = tf.cast(conv1d_1_norm,
                                  precision_np)
        
        # Second Convolution Layer
        
        conv1d_2 = tf.layers.conv1d(
                                    inputs=conv1d_1_norm16,
                                    filters=feature_maps_2,
                                    kernel_size=filter_width_2,
                                    strides=kernel_stride_2,
                                    padding="valid",
                                    activation=tf.nn.relu,
                                    name="conv1d_2"
                                    )
        conv1d_2_cast = tf.cast(conv1d_2,
                                precision_np)
        conv1d_2_norm = tf.layers.batch_normalization(conv1d_2_cast, 
                                                      training = True, 
                                                      fused=False, 
                                                      name = "bn2")
        conv1d_2_norm16 = tf.cast(conv1d_2_norm,
                                  precision_np)
        
        # Final Convolution Layer
        
        conv1d_f = tf.layers.conv1d(
                                    inputs=conv1d_2_norm16,
                                    filters=feature_maps_3,
                                    kernel_size=filter_width_3,
                                    strides=kernel_stride_3,
                                    padding="valid",
                                    activation=tf.nn.relu,
                                    name="conv1d_f"
                                    )
        conv1d_3_cast = tf.cast(conv1d_f,
                                precision_np)
        conv1d_3_norm = tf.layers.batch_normalization(conv1d_3_cast, 
                                                      training = True, 
                                                      fused=False, 
                                                      name = "bn3")
        conv1d_3_norm16 = tf.cast(conv1d_3_norm,
                                  precision_np)
        
        conv1d_flat = tf.contrib.layers.flatten(conv1d_3_norm16)
        
        # Fully Connected Layers
        
        fc_f = tf.layers.dense(
                                conv1d_flat,
                                hidden_layer_nodes_f, 
                                activation=tf.nn.relu,
                                name="logits"
                                )
        fc_f_16 = tf.cast(fc_f,precision)
        
        # Fully Connected Layers
        
        fprob = tf.nn.softmax(fc_f_16)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y_target,logits=fc_f_16)
        
        return tf.get_default_graph()
    #_____________________________________________________________
    
import_npy()
setup(test_on)

 # Placeholders
# Initialize Graph

thisgraph = generate_model (12, 8, 4);

localtime = time.asctime( time.localtime(time.time()) )
print("Graph Defined :", localtime)

# Initialize All Variables

localtime = time.asctime( time.localtime(time.time()) )
print("Initialized Variables:", localtime)

# Add ops to save and restore all the variables.

# Restore Meta graph

#saver = tf.train.import_meta_graph(meta)
#saver.restore(sess, tf.train.latest_checkpoint('../trained_models/pruned_networks/'))


init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
    
sess.run(init_g)
sess.run(init_l)

# Restore Model

saver = tf.train.Saver()
saver.restore(sess,model)

print("Model restored.")

def get_var(layer,parameter):
    varpath = str(variable_scope+"/"+layer+"/"+parameter)
    varout = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, varpath)[0])
    return varout

def l1_norm(weights):
    return np.sum(np.abs(weights),(0,1))

def prune(layer,bias,next_layer,num_prune):
    
    norms = l1_norm(layer)
    to_prune = norms.argsort()[:num_prune]
    layer_out = layer.T
    
    if(type(next_layer)==np.ndarray):
        next_layer_out = next_layer
    
    for prune_index in to_prune:
        layer_out[prune_index] = layer_out[prune_index]*0.0
        bias[prune_index] = 0
        if(type(next_layer)==np.ndarray):
            next_layer_out[:,prune_index,:] = next_layer_out[:,prune_index,:]*0.0 
    layer_out = layer_out.T
    
    if(type(next_layer)==np.ndarray):
        next_layer_out = next_layer_out.reshape(next_layer.shape)
        
    if(type(next_layer)==np.ndarray):
        return layer_out, bias, next_layer_out
    else:
        return layer_out, bias

def prune_layer(tensor,next_tensor,parameter,num_prune):
    
    varpath_k = str(variable_scope+"/"+tensor+"/"+"kernel")
    varpath_b = str(variable_scope+"/"+tensor+"/"+"bias")
    if(next_tensor):
        varpath_next = str(variable_scope+"/"+next_tensor+"/"+"kernel")
    
    np_tensor = get_var(tensor,'kernel')
    np_tensorb = get_var(tensor,'bias')
    if(next_tensor):
        np_tensor_next = get_var(next_tensor,parameter)
    
    if(next_tensor):
        np_pruned,np_pruned_bias,np_pruned_next = prune(np_tensor,np_tensorb,np_tensor_next,num_prune)
    else:
        np_pruned,np_pruned_bias = prune(np_tensor,np_tensorb,None,num_prune)
    
    sess.run(tf.assign(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, varpath_k)[0],np_pruned))
    sess.run(tf.assign(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, varpath_b)[0],np_pruned_bias))
    if(next_tensor):
        sess.run(tf.assign(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, varpath_next)[0],np_pruned_next))
    
    return 0

def inference(rx,ry,batch_size,generations):
    
    # Log vectors
    
    loss_list = []
    
    # Speeds
    
    prediction_list = []
    success_rate = []
    successful_guesses = []

    for i in range(generations):
         with thisgraph.as_default():
                
             # Get random batch
                
                rand_index = np.random.choice(len(x_vals_train), size = batch_size)
                rand_x = [x_vals_train[rand_index]]
                rand_y = y_vals_train[rand_index]
                
                temp_loss,pred_training_s = sess.run([loss,fprob], 
                                                     feed_dict={x_data : np.array([rand_x]).reshape((batch_size,sample_length,input_channels)), 
                                                                y_target:rand_y} )
        
                loss_list.append(temp_loss)
                
                guess = np.argmax(pred_training_s,axis=1)
                
                prediction_list.append(guess)
                
                correct_prediction = np.sum(np.equal(guess,rand_y))
                
                
                successful_guesses.append(correct_prediction)
                success_rate.append(correct_prediction/len(rand_y) )
                #test_accuracy = round(correct_prediction*100/len(rand_y),4)
            
    accuracy_mean = np.mean(success_rate)
    accuracy_stdev = np.std(success_rate)
    print("SPEEDS : Mean = ",round(accuracy_mean,4),
          " Std Dev = ",round(accuracy_stdev,4))
    
    return accuracy_mean
        

#test_out = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, varpath)[0])
# Store values of trained variables 
#(For analysis. These values are not used by the model directly)

trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)



X,Y = np.meshgrid(range(feature_maps+1),range(feature_maps_2+1))

ac_list = []

rand_index = np.random.choice(len(x_vals_train), size = batch_size)
rx = [x_vals_train[rand_index]]
ry = y_vals_train[rand_index]

saver = tf.train.Saver()
saver.restore(sess,model)

#print("Pruning for layer 1 : ",l1_fm)
   
#p_list = inference(rx,ry,batch_size,generations)

mean_acc = inference(rx,ry,batch_size,generations)

sess.close()