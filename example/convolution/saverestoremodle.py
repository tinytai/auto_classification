# Save+restore variables from /tmp/jnbcnn.ckpt
##b_fc1
##: [ 0.97454125  0.95274979  0.97416949 ...,  0.96728969  0.96845835
##  0.99264371]
## todo:  what is ... !!!!

from __future__ import print_function
import tensorflow as tf
import numpy as np

## Save to file
## remember to define the same dtype and shape when restore
##W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
##b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')
##
##init= tf.initialize_all_variables()
##
##saver = tf.train.Saver()
##
##with tf.Session() as sess:
##    sess.run(init)
##    save_path = saver.save(sess, "/tmp/save_net.ckpt")
##    print("Save to path: ", save_path)
##

################################################
##restore variables
##redefine the same shape and same type for your variables
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])


##W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
##b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

#not need init step

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "/tmp/jnbcnn.ckpt")
    print("W_conv1:\n", sess.run(W_conv1))
    print("b_conv1\n:", sess.run(b_conv1))
    print("W_conv2:\n", sess.run(W_conv2))
    print("b_conv2\n:", sess.run(b_conv2))
    print("W_fc1:\n", sess.run(W_fc1))
    print("b_fc1\n:", sess.run(b_fc1))
    print("W_fc2:\n", sess.run(W_fc2))
    print("b_fc2\n:", sess.run(b_fc2))
    















