import tensorflow as tf
import numpy as np
import math

# Import graph and restore weights

meta = "../../../pure_conv/full_precision_model.ckpt.meta"
model = "../../../pure_conv/full_precision_model.ckpt"

tdata = np.load('../../../training_data_3d.npy')

speeds = tdata[:,3,0]
types = tdata[:,3,1]
data = tdata[:,0:3,:]

del tdata

thisgraph = tf.Graph()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

saver = tf.train.import_meta_graph(meta)
saver.restore(sess, model)


conv1d_1_kernel = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                              'foo/conv1d_1/kernel'))[0]
conv1d_1_bias   = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                              'foo/conv1d_1/bias'))[0]
bn_1_gamma      = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                              'foo/bn1/gamma'))[0]
bn_1_beta       = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                              'foo/bn1/beta'))[0]

conv1d_2_kernel = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                              'foo/conv1d_2/kernel'))[0]
conv1d_2_bias   = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                              'foo/conv1d_2/bias'))[0]
bn_2_gamma      = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                              'foo/bn2/gamma'))[0]
bn_2_beta       = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                              'foo/bn2/beta'))[0]

conv1d_f_kernel = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                              'foo/conv1d_f/kernel'))[0]
conv1d_f_bias   = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                              'foo/conv1d_f/bias'))[0]
bn_3_gamma      = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                              'foo/bn3/gamma'))[0]
bn_3_beta       = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                              'foo/bn3/beta'))[0]

fc_kernel       = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                              'foo/logits/kernel'))[0]
fc_bias         = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                              'foo/logits/bias')) [0]
sess.close()
#------------------------------------------------------------------------------

# Functions

class TDNN(object):
    def __init__(self, inputs):
        x_vals = inputs
        original_shape = x_vals.shape
        x_vals = x_vals.reshape(-1,1600)
        means = np.mean(x_vals,axis=(1))
        means = means.reshape(27024,1)
        x_vals = x_vals-means
        x_vals = x_vals.reshape(original_shape)
        inputs = x_vals
        self.state = inputs
        #self.state = np.reshape(inputs,(np.shape(inputs)[0],np.shape(inputs)[2],np.shape(inputs)[1]))
        #print(inputs[0,0,:10])
        
    def relu(self,inputs):
        zero_matrix = np.zeros_like(inputs)
        multiplier_matrix = np.greater(inputs,zero_matrix)
        return np.multiply(inputs,multiplier_matrix)
        
    def convolution(self,filters,bias,stride,sample_no):
        in_channels = np.shape(filters)[1]
        out_channels = np.shape(filters)[2]
        ip_len = 1600
        
        out_list = []
        for j in range(0,out_channels):
            for i in range(0,in_channels):
                temp_out = []
                for k in range(0,ip_len/stride):
                    conv_filter = np.zeros_like(self.state[sample_no,i,:])
                    to_put = filters[:,i,j]
                    np.put(conv_filter,range(k,k+np.shape(to_put,stride)[0]),to_put)
                    conv_out = np.sum(np.multiply(self.state[sample_no,i,:],conv_filter))
                    conv_out = np.float16(conv_out)
                    temp_out.append(conv_out)
                #temp_out = temp_out[:1592:4]
            out_list.append(temp_out)
        out = np.asarray(out_list)+np.reshape(bias,(len(bias),1))
        out = self.relu(out)
        return out
        
tdnn = TDNN(data)

a_first_conv = tdnn.convolution(conv1d_1_kernel,conv1d_1_bias,4,0)
        