import tensorflow as tf
import numpy as np
import Image
import random
import matplotlib.pyplot as plt


saver.restore(sess, "regression.ckpt")
out = readImage2Array("/home/deeplearn/Desktop/carT/0.bmp")              
y_batch = y_conv.eval(feed_dict={x:out, keep_prob:1.0})
carConfig=readCarConfig("/home/deeplearn/Desktop/carT/result.txt")
car = carConfig[y_batch[0].argmax()]
print car
