#coding=utf-8

import math
from collections import namedtuple
import numpy as np
import tensorflow as tf
import tf_extended as tfe
from .custom_layers import *
from .ssd_common import  import *
import tensorflow.contrib.slim as slim

SSDParams=namedtuple("SSDParameters",['img_shape','num_classes','no_annotation_label','feat_layers','feat_shapes','anchor_size_bounds','anchor_sizes',
                                      'anchor_ratios','anchor_steps','anchor_offset','normalizations','prior_scaling'])

class SSDNet(object):
    default_params=SSDParams(img_shape=(512,512),num_classes=21,no_annotation_label=21,feat_layers=['block4','block7','block8','block9','block10','block11','block12'],feat_shapes=[(64,64),(32,32),(16,16),(8,8),(4,4),(2,2),(1,1)],
                             anchor_size_bounds=[0.10,0.90],anchor_sizes=[(20.48,51.2),(51.2,133.12),(133.12,215.04),(215.04,296.96),(296.96,378.88),(378.88,460.8),(460.8,542.72)],
                             anchor_ratios=[[2,.5],[2,.5,3,1/3.],[2,.5,3,1/3.],[2,.5,1,1/3.],[2,.5,3,1/3.],[2,.5],[2,.5]],
                             anchor_steps=[8,16,32,64,128,256,512],anchor_offset=0.5,normalizations=[20,-1,-1,-1,-1,-1,-1],prior_scaling=[0.1,0.1,0.2,0.2])
    def __init__(self,params=None):
        if isinstance(params,SSDParams):
            self.params=params
        else:
            self.params=SSDNet.default_params

    def net(self,input,is_training=True,dropout_keep_prob=0.5,prediction_fn=slim.softmax,reuse=None,scope="ssd_512_vgg"):
        r=ssd_net(inputs,num_classes=self.params.num_classes,feat_layers=self.params.feat_layers,anchor_sizes=self.params.anchor_sizes,anchor_ratios=self.params.anchor_ratios,normalizations=self.params.normalizations,
                  is_training=is_training,dropout_keep_prob=dropout_keep_prob,prediction_fn=prediction_fn,reuse=reuse,scope=scope)
        if use_feat_shapes:
            shapes=ssd_feat_shapes_from_net(r[0],self.params.feat_shapes)
            self.params=self.params._replace(feat_shapes=shapes)
        return r


def ssd_anchor_one_layer(img_shape,feat_shape,sizes,ratios,step,offset=0.5,dtype=np.float32):
    y,x=np.mgrid[0:feat_shape[0],0:feat_shape[1]]
    y=(y.astype(dtype)+offset)*step/img_shape[0]
    x=(x.astype(dtype)+offset)*step/img_shape[1]
    y=np.expand_dims(y,axis=-1)
    x=np.expand_dims(x,axis=-1)

    num_anchors=len(sizes)+len(ratios)
    h=np.zeros((num_anchors,),dtype=dtype)
    w=np.zeros((num_anchors,),dtype=dtype)
    h[0]=sizes[0]/img_shape[0]
    w[0]=sizes[0]/img_shape[1]
    di=1
    if len(sizes)>1:
        h[1]=math.sqrt(sizes[0]*sizes[1])/img_shape[0]
        w[1]=math.sqrt(sizes[0]*sizes[1])/img_shape[1]
        di+=1
    for i,r in enumerate(ratios):
        h[i+di]=sizes[0]/img_shape[0]/math.sqrt(r)
        w[i+di]=sizes[0]/img_shape[1]*math.sqrt(r)
    return y,x,h,w

def ssd_anchors_all_layers(img_shape,layers_shape,anchor_sizes,anchor_ratios,anchor_steps,offset=0.5,dtype=np.float32):
    layers_anchors=[]
    for i,s in enumerate(layers_shape):
        anchor_boxes=ssd_anchor_one_layer(img_shape,anchor_sizes[i],anchor_ratios[i],anchor_steps[i],offset=offset,dtype=dtype)
        layers_anchors.append(anchor_boxes)
    return layers_anchors

def ssd_net(inputs,num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope="ssd_512_vgg"):
    end_points={}
    with tf.variable_scope(scope,"ssd_512_vgg",[inputs],reuse=reuse):
        net=slim.repeat(inputs,2,slim.conv2d,64,[3,3],scope="conv1")
        end_points["block1"]=net
        net=slim.max_pool2d(net,[2,2],scope="pool1")
        net=slim.repeat(net,2,slim.conv2d,128,[3,3],scope="conv2")
        end_points["block2"]=net
        net=slim.max_pool2d(net,[2,2],scope="pool2")
        net=slim.repeat(net,3,slim.conv2d,256,[3,3],scope="conv3")
        end_points["block3"]=net
        net=slim.max_pool2d(net,[2,2],scope="pool3")
        net=slim.repeat(net,3,slim.conv2d,512,[3,3],scope="conv4")
        end_points["block4"]=net
        net=slim.max_pool2d(net,[2,2],scope="pool4")
        net=slim.repeat(net,3,slim.conv2d,512,[3,3],scope="conv5")
        end_points["block5"]=net
        net=slim.max_pool2d(net,[3,3],1,scope='pool5')

        net=slim.conv2d(net,1024,[3,3],rate=6,scope="conv6")
        end_points["block6"]=net
        net=slim.conv2d(net,1024,[1,1],scope="conv7")
        end_points["block7"]=net

        end_point="block8"
        with tf.variable_scope(end_point):
            net=slim.conv2d(net,256,[1,1],scope="conv1*1")
            net=pad2d(net,pad=[1,1])
            net=slim.conv2d(net,512,[3,3],stride=2,scope="conv3*3",padding="VALID")

        end_points[end_point]=net
        end_point="block9"
        with tf.variable_scope(end_point):
            net=slim.conv2d(net,128,[1,1],scope="conv1*1")
            net=pad2d(net,pad=(1,1))
            net=slim.conv2d(net,256,[3,3],stride=2,scope="conv3*3",padding="VALID")

        end_points[end_point]=net
        end_point="block10"
        with tf.variable_scope(end_point):
            net=slim.conv2d(net,128,[1,1],scope="conv1*1")
            net=pad2d(net,pad=(1,1))
            net=slim.conv2d(net,256,[3,3],stride=2,scope="conv3*3",padding="VALID")

        end_points[end_point]=net
        end_point="block11"
        with tf.variable_scope(end_point):
            net=slim.conv2d(net,128,[1,1],scope="conv1*1")
            net=pad2d(net,pad=(1,1))
            net=slim.conv2d(net,256,[3,3],stride=2,scope="cnv3*3",padding="VALID")

        end_points[end_point]=net
        end_point="block12"
        with tf.variable_scope(end_point):
            net=slim.conv2d(net,128,[1,1],scope="conv1*1")
            net=pad2d(net,pad=(1,1))
            net=slim.conv2d(net,256,[4,4],scope="conv4*4",padding="VALID")

        end_points[end_point]=net

        predictions=[]
        logits=[]
        localisations=[]
        for i,layer in enumerate(feat_layers):
            with tf.varaible_scope(layer+"_bbox"):
                p,l=ssd_multibox_layer(end_points[layer],num_classes,anchor_sizes[i],anchor_ratios[i],normalizations[i])
                predictions.append(prediction_fn(p))
                logits.append(p)
                localisations.apepnd(l)

            return predictions,localisations,logits,end_point

ssd_net.default_image_size=512

def ssd_arg_scope(weight_decay=0.0005,data_format="NHWC"):
    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weight_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initialzier=tf.contrib.layers.xavier_initializer(),
                        bias_initializer=tf.zeros_intializer()):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d],padding="SAME",data_format=data_format):
            with slim.arg_scope([pad2d],l2_normalization,channel_to_last,data_format=data_format) as sc:
                return sc

def ssd_multibox_layer(inputs,num_classes,sizes,ratios=[1],normalization=-1,bn_normalization=False):
    net=inputs
    if normalization>0:
        net=l2_normalization(net,scaling=True)
    num_anchors=len(sizes)+len(ratios)

    num_loc_pred=num_anchors*4
    loc_pred=slim.conv2d(net,num_loc_pred,[3,3],activation_fn=None,scope="conv_loc")
    loc_pred=channel_to_last(loc_pred)
    loc_pred=tf.reshape(loc_pred,tensor_shape(loc_pred,4)[:-1]+[num_anchors,4])

    num_cls_pred=num_anchors*num_classes
    cls_pred=slim.con2d(net,num_cls_pred,[3,3],activation_fn=None,scope="conv_cls")
    cls_pred=channel_to_last(cls_pred)
    cls_pred=tf.reshape(cls_pred,tensor_shape(cls_pred,4)[:-1]+[num_anchors,num_classes])

    return cls_pred,loc_pred
