import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

test_on = "types" # speeds or types

variable_scope = "foo" #/ "pruned"
model = "../trained_models//quantized/quantized_8_speeds.ckpt"
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
 
generations         = 20
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
    global logit_size,saver
    
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

#______________________________________________________________________________
    
import_npy()
setup(test_on)

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
    print(tf.size(conv1d_flat))
    
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

# Initialize Graph

thisgraph = tf.get_default_graph()

localtime = time.asctime( time.localtime(time.time()) )
print("Graph Defined :", localtime)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y_target,logits=fc_f_16)

# Initialize All Variables

localtime = time.asctime( time.localtime(time.time()) )
print("Initialized Variables:", localtime)

# Log vectors

loss_list = []
test_loss = []

# Speeds

test_predictions = []
prediction_list = []
success_rate = []
successful_guesses = []
test_success_rate = []    

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

varpath = str(variable_scope+"/logits/kernel")
test_out = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, varpath)[0])
# Store values of trained variables 
#(For analysis. These values are not used by the model directly)

trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

for i in trainable_variables:
    print(i.name)
    
#test_var_b = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'pruned/conv1d_2/bias')[0])
#test_var_k = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv1d_2_pruned/kernel')[0])

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
            test_accuracy = round(correct_prediction*100/len(rand_y),4)
            
accuracy_mean = np.mean(success_rate)
accuracy_stdev = np.std(success_rate)
print("SPEEDS : Mean = ",round(accuracy_mean,4),
      " Std Dev = ",round(accuracy_stdev,4))

sess.close()

# Plot values

plt.plot(success_rate)
plt.title('Accuracies')
plt.show()

def plot_graph(weight_data,title):
    plt.hist(weight_data.flatten(),15)
    plt.title(title)
    plt.show()
    
#plot_graph(z_conv1d_1_kernel, "1st Convolution Layer Weights")
#plot_graph(z_conv1d_1_bias, "1st Convolution Layer Bias")
#plot_graph(z_conv1d_2_kernel, "2nd Convolution Layer Weights")
#plot_graph(z_conv1d_2_bias, "2nd Convolution Layer Bias")
#plot_graph(z_conv1d_f_kernel, "3rd Convolution Layer Weights")
#plot_graph(z_conv1d_f_bias, "3rd Convolution Layer Bias")
#plot_graph(z_logits_kernel, "Softmax Layer Weights")