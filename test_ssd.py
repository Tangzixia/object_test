#coding=utf-8

import tensorflow as tf

i=tf.get_variable("ii",dtype=tf.int32,shape=[],initializer=tf.ones_initializer())
n=tf.constant(10)

def cond(a,n):
    return a<n
def body(a,n):
    a=a+1
    return a,n

a,n=tf.while_loop(cond,body,[a,n])
init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    res=sess.run([a,n])
    print(res)