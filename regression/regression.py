
"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

##def add_layer(inputs, in_size, out_size, activation_function=None):
##    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
##    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
##    Wx_plus_b = tf.matmul(inputs, Weights) + biases
##    if activation_function is None:
##        outputs = Wx_plus_b
##    else:
##        outputs = activation_function(Wx_plus_b)
##    return outputs

# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
#print(noise)
y_data = np.square(x_data) - 0.5 + noise
#print (y_data)

#plt.scatter(x_data, y_data)
#plt.show()

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer
#l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
Weights1 = tf.Variable(tf.random_normal([1, 10], name="W"))
biases1 = tf.Variable(tf.zeros([1, 10]) + 0.1, name="B")
Wx_plus_b1 = tf.matmul(xs, Weights1) + biases1
l1 = tf.nn.relu(Wx_plus_b1)


# add output layer
#prediction = add_layer(l1, 10, 1, activation_function=None)
Weights2 = tf.Variable(tf.random_normal([10, 1]))
biases2 = tf.Variable(tf.zeros([1, 1]) + 0.1)
prediction = tf.matmul(l1, Weights2) + biases2
#prediction  = Wx_plus_b2

# the error between prediciton and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# important step
init = tf.initialize_all_variables()
saver = tf.train.Saver()
sess= tf.Session()
sess.run(init)

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

print ("please input 1 for traning 2 for test!")
flag = raw_input(">")

if flag == "1":
    for i in range(1000):
        # training
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    #feed_dict={xs: x_data, ys: y_data}
    #print (feed_dict)
        if i % 50 == 0:
            # to see the step improvement
    #       print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    #       print(prediction)
            # to visualize the result and improvement
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            # plot the prediction
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)
            time.sleep(1)
    
    print("Weights1=\n",sess.run(Weights1))
    print("biases1=\n",sess.run(biases1))
    print("Weights2=\n",sess.run(Weights2))
    print("biases2=\n",sess.run(biases2))
    
    save_path = saver.save(sess, "regression.ckpt")
    print ("Model saved in file: ", save_path)
else:
    saver.restore(sess, "/home/loony/work/tenserflow/example/regression/regression.ckpt")
#prediction_value = sess.run(prediction, feed_dict={xs: x_data})
    try:
        ax.lines.remove(lines[0])
    except Exception:
            pass
    prediction_value = sess.run(prediction, feed_dict={xs: x_data})
    # plot the prediction
    lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
    plt.pause(0.1)
    raw_input("End the APP by entern any key!")


