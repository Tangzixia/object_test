import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm,flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np

weight_decay=0.0005
momentum=0.9
init_learning_rate=0.1
reduction_ratio=4

batch_size=128
iteration=391
test_iteration=10
total_epochs=100

def conv_layer(input,filter,kernel,stride,padding="SAME",layer_name="conv",activation=True):
    with tf.name_scope(layer_name):
        network=tf.layers.conv2d(inputs=input,use_bias=True,filters=filter,kernel_size=kernel,strides=stride,padding=padding)
        if activation:
            network=Relu(network)
        return network
def Fully_connected(x,units=class_num,layer_name="fully_connected"):
    with tf.name_scope(layer_name):
        return tf.layers.dense(inputs=x,use_bias=True,units=units)

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)

def Global_Average_Pooling(x):
    return global_avg_pool(x,name="Global_avg_pooling")

def Max_pooling(x,pool_size=[3,3],stride=2,padding="VALID"):
    return tf.layers.max_pooling2d(inputs=x,pool_size=pool_size,strides=str,padding=padding)

def Avg_pooling(x,pool_size=[3,3],stride=1,padding="SAME"):
    return tf.layers.average_pooling2d(inputs=x,pool_size=pool_size,strides=stride,padding=padding)

def Batch_Normalization(x,training,scope):
    with arg_scope([batch_norm],scope=scope,updates_collections=None,decay=0.9,center=True,scale=True,zero_bias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(input=x,is_training=training,reuse=None),
                       lambda: batch_norm(input=2,is_training=training,reuse=True))

def Concatenation(layers):
    return tf.concat(layers,axis=3)

def Dropout(x,rate,training):
    return tf.layers.dropout(input=x,rate=rate,training=training)


class SE_Inception_resnet_v2():
    def __init__(self,x,training):
        self.training=training
        self.model=self.Build_SEnet(x)

    def Stem(self,x,scope):
        with tf.name_scope(scope):
            x=conv_layer(x,filter=32,kernel=[3,3],stride=2,padding="VALID",layer_name=scope+"_conv1")
            x=conv_layer(x,filter=32,kernel=[3,3],padding="VALID",layer_name=scope+"_conv2")
            block_1=conv_layer(x,filter=64,kernel=[3,3],layer_name=scope+"_conv3")

            split_max_x=Max_pooling(block_1)
            split_conv_x=conv_layer(block_1,filter=96,kernel=[3,3],stride=2,padding="VALID",layer_name=scope+"_split_conv1")
            x=Concatenation([split_max_x,split_conv_x])

            split_conv_x1=conv_layer(x,filter=64,kernel=[1,1],layer_name=scope+"_split_conv2")
            split_conv_x1=conv_layer(split_conv_x1,filter=96,kernel=[3,3],padding="VALID",layer_name=scope+"_split_conv3")

            split_conv_x2=conv_layer(split_conv_x1,filter=64,kernel=[1,1],layer_name=scope+"_split_conv4")
            split_conv_x2=conv_layer(split_conv_x2,filter=64,kernel=[7,1],layer_name=scope+"_split_conv5")
            split_convx2=conv_layer(split_conv_x2,filter=64,kernel=[1,7],layer_name=scope+"_split_conv6")
            split_conv_x2=conv_layer(split_conv_x2,filter=96,kernel=[3,3],padding="VALID",layer_name=scope+"_split_conv7")

            x=Concatenation([split_conv_x1,split_conv_x2])
            split_conv_x=conv_layer(x,filter=192,kernel=[3,3],stride=2,padding="VALID",layer_name=scope+"_split_conv8")
            split_max_x=Max_pooling(x)

            x=Concatenation([split_conv_x,split_max_x])
            x=Batch_Normalization(x,training=self.training,scope=scope+"_batch1")
            x=Relu(x)

        return x

    def Reduction_A(self,x,scope):
        with tf.name_scope(scope):
            k=256
            l=256
            m=384
            n=384

            split_max_x=Max_pooling(x)
            split_conv_x1=conv_layer(x,filter=n,kernel=[3,3],stride=2,padding="VALID",layer_name=scope+"_split_conv1")
            split_conv_x2=conv_layer(x,filter=k,kernel=[1,1],layer_name=scope+"_split_conv2")
            split_conv_x2=conv_layer(split_conv_x2,filter=m,kernel=[3,3],stride=2,padding="VALID",layer_name="split_conv3")

            x=Concatenation([split_max_x,split_conv_x1,split_conv_x2])
            x=Batch_Normalization(x,training=self.training,scope=scope+"_batch1")
            x=Relu(x)

            return x

    def Squeeze_excitation_layer(self,input_x,out_dim,ratio,layer_name):
        with tf.name_scope(layer_name):
            squeeze=Global_Average_Pooling(input_x)

            excitation=Fully_connected(squeeze,units=out_dim/ratio,layer_name=layer_name+"_fully_connected1")
            excitation=Relu(excitation)
            excitation=Fully_connected(excitation,units=out_dim,layer_name=layer_name+"_fully_connected2")
            excitation=Sigmoid(excitation)

            excitation=tf.reshape(excitation,[-1,1,1,out_dim])
            scale=input_x*excitation
            return scale


    def Inception_resnet_A(self,x,scope):
        with tf.name_scope(scope):
            init=x

            split_conv_x1=conv_layer(x,filter=32,kernel=[1,],layer_name=scope+"_split_conv1")
            split_conv_x2=conv_layer(split_conv_x1,filter=32,kernel=[1,1],layer_name=scope+"_split_conv2")
            split_conv_x2=conv_layer(split_conv_x2,filter=32,kernel=[3,3],layer_name=scope+"_split_conv3")

            split_conv_x3 = conv_layer(x, filter=32, kernel=[1,1], layer_name=scope+'_split_conv4')
            split_conv_x3 = conv_layer(split_conv_x3, filter=48, kernel=[3,3], layer_name=scope+'_split_conv5')
            split_conv_x3 = conv_layer(split_conv_x3, filter=64, kernel=[3,3], layer_name=scope+'_split_conv6')

            x=Concatenation([split_conv_x1,split_conv_x2,split_conv_x3])
            x=conv_layer(x,filter=384,kernel=[1,1],layer_name=scope+"_final_conv1",activation=False)

            x= x*0.1
            x=init+x

            x=Batch_Normalization(x,training=self.training,scope=scope+"_batch1")
            x=Relu(x)
            return x

