# -*- coding: utf-8 -*-
# @Time    : 2018/6/29 下午6:11
# @Author  : Xieli Ruan
# @Site    : 
# @File    : mofan4.py
# @Software: PyCharm

# rnn
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# this is data
mnist =input_data.read_data_sets('MNIST_data',one_hot=True)

# hyperparameters
lr=0.001 #learning rate
tranining_iters=100000 #迭代次数
batch_size=128#


n_inputs=28 #MNIST data input(img shape:28*28)28行
n_steps=28 #time steps每行28个
n_hidden_unis=128 #neurons in hidden layer
n_classes=10 #MNIST classes(0-9 digits)

# tf Graph input
x=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y=tf.placeholder(tf.float32,[None,n_classes])

# Define weights
weights={
    # (28,128)
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_unis])),
    # (128,10)
    'out':tf.Variable(tf.random_normal([n_hidden_unis,n_classes]))
}

biases={
    # (128)
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_unis,])),
    # (10)
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

def RNN(X,weights,biases):
    # hidden layer for input to cell
    ##########################################################
    # X(128batch, 28 steps,28inputs)
    # X==>(128*28, 28 inputs)
    X=tf.reshape(X,[-1,n_inputs])

    # X_in==>(128 batch * 28 steps, 128 hidden)
    X_in=tf.matmul(X,weights['in'])+biases['in']
    X_in=tf.reshape(X_in,[-1,n_steps,n_hidden_unis])

    # cell
    ##########################################################
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis,forget_bias=1.0,state_is_tuple=True)
    # lstm cell is divided into 2 parts(c_state,m_state)
    _init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)

    # time_major：因为n_steps在第二个位置，次要维度
    outputs,states=tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)



    # hidden layer for output as the final result
    ##########################################################
    # states[0]是主线，states[1]是分线=output[-1]
    results=tf.matmul(states[1],weights['out'])+biases['out']

    return results

pred=RNN(x,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=pred,logits=y))
train_op=tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init =tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step=0
    while step*batch_size<tranining_iters:
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        batch_xs=batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run([train_op],feed_dict={
            x:batch_xs,
            y:batch_ys
        })
        if step%20==0:
            print(sess.run(accuracy,feed_dict={
                x:batch_xs,
                y:batch_ys
            }))