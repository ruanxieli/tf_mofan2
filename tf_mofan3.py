# -*- coding: utf-8 -*-
# @Time    : 2018/6/28 下午5:02
# @Author  : Xieli Ruan
# @Site    : 
# @File    : tf_mofan3.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

def add_layer(inputs,in_size,out_size,layer_name,activation_function=None):
    # add one more layer and return the output of this layer
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases

    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result



# define placeholder for inputs to network
# xs：数据类型是float32，格式None（不规定有多少sample），每个sample大小是784（784个像素点）
xs=tf.placeholder(tf.float32,[None,784])
# ys：每个sample有10个输出
ys=tf.placeholder(tf.float32,[None,10])

# add output layer
l1=add_layer(xs,784,100,'l1',activation_function=tf.nn.tanh)
prediction=add_layer(l1,100,10,'l2',activation_function=tf.nn.softmax)

# loss
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                            reduction_indices=[1])) #loss

train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

# important step



sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50==0:

        print(compute_accuracy(mnist.test.images,mnist.test.labels))

