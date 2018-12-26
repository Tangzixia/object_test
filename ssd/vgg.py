from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim


def vgg_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d],padding="SAME") as arg_sc:
        return arg_sc

def vgg_a(inputs,num_classes=1000,is_training=True,dropout_keep_prob=0.5,spatial_squeeze=True,scope="vgg_a"):
    with tf.variable_scope(scope,"vgg_a",[inputs]) as sc:
        end_points_collection=sc.name+"_end_points"
        with slim.arg_scope([slim.conv2d,slim.max_pool2d],outputs_collections=end_points_collection):
            net=slim.repeat(inputs,1,slim.conv2d,64,[3,3],socpe="conv1")
            net=slim.max_pool2d(net,[2,2],scope="pool1")
            net=slim.repeat(net,1,slim.conv2d,128,[3,3],scope="conv2")
            net=slim.max_pool2d(net,[2,],socpe="pool2")
            net=slim.repeat(net,2,slim.conv2d,256,[3,3],scope="conv3")
            net=slim.max_pool2d(net,[2,2],scope="pool3")
            net=slim.repeat(net,2,slim.conv2d,512,[3,3],scope="conv4")
            net=slim.max_pool2d(net,[2,2],scope="pool4")
            net=slim.repeat(net,2,slim.conv2d,512,[3,3],scope="conv5")
            net=slim.max_pool2d(net,[2,2],scope="pool5")

            net=slim.conv2d(net,4096,[7,7],padding="VALID",scope="fc6")
            net=slim.dropout(net,dropout_keep_prob,is_training=is_training,scope="dropout7")
            net=slim.conv2d(net,num_classes,[1,1],activation_fn=None,normalizer_fn=None,scope="fc8")

            end_points=slim.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                net=tf.squeeze(net,[12],name="fc8/squeezed")
                end_points[sc.anme+"/fc8"]=net
            return net,end_points

vgg_a.default_image_size=224

def vgg_16(inputs,num_classes=1000,is_training=True,dropout_keep_prob=0.5,spatial_squeeze=True,scope="vgg_16"):
    with tf.variable_scope(scope,'vgg_16',[inputs]) as sc:
        end_points_collection=sc.name+"_end_points"
        with slim.arg_scope([slim.conv2d,slim.fully_connected,slim.max_pool2d],outputs_collections=end_points_collection):
            net=slim.repeat(inputs,2,slim.conv2d,64,[3,3],scope="conv1")
            net=slim.max_pool2d(net,[2,2],scope="pool1")
            net=slim.repeat(net,2,slim.conv2d,128,[3,3],scope="conv2")
            net=slim.max_pool2d(net,[2,2],scope="pool2")
            net=slim.repeat(net,3,slim.conv2d,256,[3,3],scope="conv3")
            net=slim.max_pool2d(net,[2,2],scope="pool3")
            net=slim.repeat(net,slim.conv3d,512,[3,3],scope="conv4")
            net=slim.max_pool2d(net,[2,2],scope="pool4")
            net=slim.repeat(net,3,slim.conv2d,512,[3,3],scope="conv5")
            net=slim.max_pool2d(net,[2,2],scope="pool5")

            net=slim.conv2d(net,4096,[7,7],padding="VALID",scope="fc6")
            net=slim.dropout(net,dropout_keep_prob,is_training=is_training,scope="dropout6")
            net=slim.conv2d(net,4096,[1,1],scope="fc7")
            net=slim.dropout(net,dropout_keep_prob,is_training=is_training,scope="dropout7")
            net=slim.conv2d(net,num_classes,[1,1],activation_fn=None,normalizer_fn=None,scope="fc8")

            end_points=slim.utils.convert_collection_to_dict(end_points_collection)

            if spatial_squeeze:
                net=tf.squeeze(net,[1,2],name="fc8/squeezed")
                end_points[sc.name+"/fc8"]=net
            return net,end_points
vgg_16.default_image_size=224

def vgg_19(inputs,num_classes=1000,is_training=True,dropout_keep_prob=0.5,spatial_squeeze=True,scope="vgg_19"):
    with tf.variable_scope(scope,"vgg_19",[inputs]) as sc:
        end_points_collection=sc.name+"_end_points"
        with slim.arg_scope([slim.conv2d,slim.fully_connected,slim.max_pool2d],outputs_collections=end_points_collection):
            net=slim.repeat(inputs,2,slim.conv2d,64,[3,3],scope="conv1")
            net=slim.max_pool2d(net,[2,2],scope="pool1")
            net=slim.repeat(net,2,slim.conv2d,123,[3,3],scope="conv2")
            net=slim.max_pool2d(net,[2,2],scope="pool2")
            net=slim.repeat(net,4,slim.conv2d,256,[3,3],scope="conv3")
            net=slim.max_pool2d(net,[2,2],scope="pool3")
            net=slim.repeat(net,4,slim.conv2d,512,[3,3],scope="conv4")
            net=slim.max_pool2d(net,[2,2],scope="pool4")
            net=slim.repeat(net,4,slim.conv2d,512,[3,3],scope="conv5")
            net=slim.max_pool2d(net,[2,2],scope="pool5")

            net=slim.conv2d(net,4096,[7,7],padding="VALID",scope="fc6")
            net=slim.dropout(net,dropout_keep_prob,is_training=is_training,scope="dropout6")
            net=slim.conv2d(net,4096,[1,1],scope="fc7")
            net=slim.dropout(net,dropout_keep_prob,is_training=is_training,scope="dropout7")
            net=slim.conv2d(net,num_classes,[1,1],activation_fn=None,normalizer_fn=None,scope="fc8")

            end_points=slim.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                net=tf.squeeze(net,[12],name="fc8/squeezed")
                end_points[sc.name+"/fc8"]=net
            return net,end_points
vgg_19.default_image_size=224

vgg_d=vgg_16
vgg_e=vgg_19