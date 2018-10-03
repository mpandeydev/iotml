import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

meta = "../../pure_conv/full_precision_model.ckpt.meta"
model = "../../pure_conv/full_precision_model.ckpt"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)

saver = tf.train.import_meta_graph(meta)
graph = tf.get_default_graph()

test_var = graph.get_tensor_by_name('foo/conv1d_1/kernel:0')*0

saver.restore(sess, model)
run_test = sess.run(test_var)
