# -*- coding: utf-8 -*-
# @Time    : 2018/5/28 下午6:50
# @Author  : Xieli Ruan
# @Site    : TensorBoard
# @File    : tf_mofan2.py
# @Software: PyCharm

import tensorflow as tf
""" 
为了在TensorBoard中展示节点名称，设计网络时会常使用tf.name_scope限制命名空间， 
在这个with下所有的节点都会自动命名为input/xxx这样的格式。 
"""
def add_layer(inputs, in_size, out_size,activation_function=None):
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='W')

        with tf.name_scope('biases'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='b')

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(inputs,Weights)+biases
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        return outputs

# define palceholder for inputs to network
with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')



#hiden layer1
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)

# output layer
prediction=add_layer(l1,10,1,activation_function=None)

# the error between prediction and real data
with tf.name_scope('loss'):
    # reduction_indices=[1]：按行求和
    loss= tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))

with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init=tf.global_variables_initializer()
sess=tf.Session()
writer=tf.summary.FileWriter('logs/',sess.graph)

sess.run(init)
'''
jjdemacbook-pro:tf_mofan2 jj$ tensorboard --logdir='/Users/jj/Virtualenv/tensorflow-env/tf_mofan2/logs'
使用了完整路径
'''
