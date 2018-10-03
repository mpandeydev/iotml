import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

x_data = tf.placeholder(shape=(None, sample_length), dtype=tf.float32)
y_target = tf.placeholder(shape=(None), dtype=tf.int32)

#*****************************************************************************
# Change functions from here on out if architecture changes

#filterz = tf.random_normal(shape=[3], dtype=tf.float32)
#filterz = tf.expand_dims(filterz, axis = 2)
#conv2 = tf.nn.conv1d(x_data,filterz, stride=2, padding="VALID")
filter_1d = np.array(np.random.rand(3))
filter_1d = tf.convert_to_tensor(filter_1d,dtype=np.float32)

filter_ip = tf.reshape(x_data,shape=[tf.shape(x_data)[0],sample_length,1])
filter_1d = tf.reshape(filter_1d,[3,1,1])
#dense = tf.layers.dense(inputs=conv2, units=1024, activation=tf.nn.relu)

A1 = tf.Variable(tf.random_normal(shape=[2399,hidden_layer_nodes]))
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))

A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,hidden_layer_nodes_2]))
b2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_2]))

A3 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_2,hidden_layer_nodes_3]))
b3 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_3]))

A4 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_3,hidden_layer_nodes_4]))
b4 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_4]))

A5 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_4,hidden_layer_nodes_5]))
b5 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_5]))

A6 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_5,hidden_layer_nodes_6]))
b6 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_6]))

A7 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_6,hidden_layer_nodes_7]))
b7 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_7]))

Af = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_7,hidden_layer_nodes_f]))
bf = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes_f]))

# Layer outputs 
# relu
# sigmoid
 
# Set up Computation Graph 
conv2 = tf.squeeze(tf.nn.conv1d(value=filter_ip,filters=filter_1d,stride=kernel_stride,padding="VALID"))
#conv2 = tf.reshape(conv2,[1,tf.shape(x_data)[0]])
hidden_out = activation(conv2,A1,b1)
hidden_out_2 = activation(hidden_out,A2,b2)
hidden_out_3 = activation(hidden_out_2,A3,b3)
hidden_out_4 = activation(hidden_out_3,A4,b4)
hidden_out_5 = activation(hidden_out_4,A5,b5)
hidden_out_6 = activation(hidden_out_5,A6,b6)
hidden_out_7 = activation(hidden_out_6,A7,b7)
final_out = activation(hidden_out_7,Af,bf)
fprob = tf.nn.softmax(final_out, name=None)
 
#*****************************************************************************

# Define Loss function

loss = tf.losses.sparse_softmax_cross_entropy(labels=y_target,logits=final_out)
#loss = tf.losses.softmax_cross_entropy(onehot_labels=y_target,logits=final_out)

# Optimizer function

my_opt = tf.train.AdagradOptimizer(step_size)
#my_opt = tf.train.AdagradOptimizer(step_size)
train_step = my_opt.minimize(loss)

# Initialize all variables

init = tf.initialize_all_variables()
sess.run(init)

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
    
    # Get random batch
    
    rand_index = np.random.choice(len(x_vals_train), size = batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = y_vals_train[rand_index]
    
    # Training step
    
    sess.run(train_step, feed_dict={x_data : rand_x, y_target:rand_y})
    temp_loss,get_pred = sess.run([loss,fprob], feed_dict={x_data : rand_x, y_target:rand_y} )
    loss_vec.append(temp_loss)
    
    temp_loss,get_pred = sess.run([loss,fprob], feed_dict={x_data : x_vals_train, y_target:y_vals_train} )
    guess = np.argmax(get_pred,axis=1)
    pred_vec.append(guess)
    correct_pred = np.sum(np.equal(guess,y_vals_train))
    successful_guesses.append(correct_pred)
    success_rate.append(correct_pred/len(x_vals_train) )
    train_accuracy = round(correct_pred*100/len(x_vals_train),4)
    
    # Get testing loss
    
    test_temp_loss,predict = sess.run([loss,fprob], feed_dict={x_data : x_vals_test, y_target:y_vals_test})
    test_loss.append(test_temp_loss)
    test_pred.append(predict)
    test_guess = np.argmax(predict,axis=1)
    test_correct_pred = np.sum(np.equal(test_guess,y_vals_test))
    test_rate.append(test_correct_pred/len(y_vals_test))
    test_accuracy = round(test_correct_pred*100/len(y_vals_test),4)
    
    # Print updates
    if (i+1)%100==0:
        #print('Generation: ' + str(i+1) + '. Loss = ' + str((temp_loss))+'. Test Loss = ' + str((test_temp_loss))+'. Test Accuracy = ' + str((test_accuracy)))
        #print('Generation: ' + str(i+1) + '. Loss = ' + str((temp_loss))+". Accuracy "+str(correct_pred*100/batch_size)+"%")
        print('Generation: ' + str(i+1) + '. Training Accuracy = ' + str((train_accuracy))+'. Test Accuracy = ' + str((test_accuracy)))
    
validation_loss,validation_predict = sess.run([loss,fprob], feed_dict={x_data : x_vals_validation, y_target:y_vals_validation})
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