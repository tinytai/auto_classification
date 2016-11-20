# -*- coding: utf-8 -*-
'''
    made by hc
    2016-10-7
'''

import tensorflow as tf
import numpy as np
import Image
import random
import matplotlib.pyplot as plt


# 待输入的占位符,x为图像数据的维度，y为分类数据的维度
# 留一个Question 以数组进行的分类目测可行了，和以数字进行的分类训练呢？why?
x = tf.placeholder("float", shape=[None, 164, 164, 1])
y_ = tf.placeholder("float", shape=[None, 66])
keep_prob = tf.placeholder("float")

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def read_train_model():
    #载入data
    trainDataPath = "/home/deeplearn/Desktop/carT/data.dt"
    print "Loading...\n"+trainDataPath + "\n"
    #载入数据与result
    srcFile=open(trainDataPath)
    srcArr=srcFile.read(-1)
    srcArr=srcArr.split(' ')
    srcArr=np.asarray(srcArr,dtype=np.float32)
    srcArr=srcArr.reshape((66,164,164,1))
    #testsrcArr=srcArr[1,45,45,0]

    #print testsrcArr
    zeroArray = np.zeros((66,66))
    print "zero Array"
    print zeroArray
    for nI in range(0,66):
        zeroArray[nI][nI] = 1

    print zeroArray
    return srcArr,zeroArray


# Network Parameters
n_input = 164 * 164 * 1# 输入数据的维数
n_classes = 66         # 标签维度
dropout = 0.75         # Dropout, probability to keep units

 
#def model():

#    return y_conv, rmse

def save_model(saver,sess,save_path):
    path = saver.save(sess, save_path)
    print 'model save in :{0}'.format(path)

def readImage2Array(path):
    im = Image.open(path)
    ims = im.resize((164,164))
    ims = ims.convert("L")
    out=np.asarray(ims)
    out=np.float32(out/255.0)
    print(out.shape)
    out=out.reshape((1,164,164,1)) 
    return out

def readCarConfig(path):
    fo=open(path,'r')
    strs=fo.read(-1)
    array = strs.split('\r\n')
    resultArr = []
    for nIndex in range(66):
        tempArr=array[nIndex].split(' ')
        tempstr=tempArr[1]
        f=tempstr.decode("gb2312")
        resultArr.append(f)
    fo.close()
    return resultArr



if __name__ == '__main__':
    sess         = tf.InteractiveSession()
##    y_conv, rmse = model()
####
#input 1 pic, 3x3, out 16
    W_conv1 = weight_variable([3, 3, 1, 16])     
    b_conv1 = bias_variable([16])
#input 16 pic, 3x3, out 32
    W_conv2 = weight_variable([2, 2, 16, 32])            
    b_conv2 = bias_variable([32])

    W_conv3 = weight_variable([3, 3, 32, 64])             
    b_conv3 = bias_variable([64])

    W_conv4 = weight_variable([2, 2, 64, 128])            
    b_conv4 = bias_variable([128])

    W_conv5 = weight_variable([2, 2, 128, 256])            
    b_conv5 = bias_variable([256])

    W_fc1 = weight_variable([4 * 4 * 256, 256])  
    b_fc1 = bias_variable([256])

    W_fc2 = weight_variable([256, 256])
    b_fc2 = bias_variable([256])
#final out 66
    W_fc3 = weight_variable([256, 66])
    b_fc3 = bias_variable([66])
    
    print(1)
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    print(2)
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)

    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    h_pool5 = max_pool_2x2(h_conv5)
    
    print(3)
    h_pool4_flat = tf.reshape(h_pool5, [-1, 4 * 4 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
    print(4)

    print(5)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    print(6)

    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    rmse = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
####
    train_step   = tf.train.AdamOptimizer(1e-3).minimize(rmse)
    sess.run(tf.initialize_all_variables())

    current_epoch = 0
    train_index = range(66)
    random.shuffle(train_index)

    X_train,y_train=read_train_model()
    print "loading ok... start training"

    saver = tf.train.Saver()
    best_validation_loss=1.0
    print 'begin training..., train dataset size:{0}'.format(1521)

    print ("please input 1 for traning 2 for test!")
    flag = raw_input(">")

if flag == "1":
    for i in xrange(400):
        random.shuffle(train_index)
        if(i < 350):
            train_step.run(feed_dict={x:X_train[train_index[0:10]],y_:y_train[train_index[0:10]], keep_prob:0.95})

        if(i >= 350):
            #print sess.run(y_conv, feed_dict={x:X_train[train_index[0:1]],y_:y_train[train_index[0:1]], keep_prob:0.95})
            #print sess.run(rmse)
            train_step.run(feed_dict={x:X_train[train_index],y_:y_train[train_index], keep_prob:0.95})
        if(i>0):#i%100 == 0
            train_loss = rmse.eval(feed_dict={x:X_train, y_:y_train, keep_prob: 0.95})
            print 'epoch {0} done! validation loss:{1}'.format(i, train_loss*100.0)
        else:
            print 'traing ...' + str(i)

    save_path = saver.save(sess, "regression.ckpt")
else:
    saver.restore(sess, "regression.ckpt")
    for i in xrange(2):
        out = readImage2Array("/home/deeplearn/Desktop/carT/"+str(i)+".bmp")
        print("/home/deeplearn/Desktop/carT/"+str(i)+".bmp")            
        y_batch = y_conv.eval(feed_dict={x:out, keep_prob:1.0})
        print y_batch[0]
        carConfig=readCarConfig("/home/deeplearn/Desktop/carT/result.txt")
        car = carConfig[y_batch[0].argmax()]
        print car




