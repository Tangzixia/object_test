#coding=utf-8

import tensorflow as tf

x=tf.zeros((2,3),name="x")
y=tf.zeros((5,3),name="y")
z=tf.concat((x,y),axis=0,name="z")

with tf.Session() as sess:
    zz=sess.run(z)
    print(zz)