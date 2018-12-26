#coding=utf-8

import math
from collections import namedtuple
import numpy as np
import tensorflow as tf
import tf_extended as tfe
from custom_layers import *
from ssd_common import *
import tensorflow.contrib.slim as slim

SSDParams=namedtuple("SSDParameters",[
    'img_shape','num_classes','feat_layers','feat_shapes','anchor_size_bounds',
    'anchor_sizes','anchor_ratios','anchor_steps','anchor_offset','normalizations','prior_scaling',
])

class SSDNet(object):
    default_params=SSDParams(img_shape=(300,300),num_classes=21,no_annotation_label=21,feat_layers=['block4','block7','block8','block9','block10','block11'],
                             feat_shapes=[(38,38),(19,19),(10,10),(5,5),(3,3),(1,1)],anchor_size_bounds=[0.15,0.90],anchor_sizes=[(21.,45.),(45.,99.),(99.,153.),(153.,207.),(207.,261.),(261.,315.)],
                             anchor_ratios=[[2,.5],[2,.5,3,1/3.],[2,.5,3,1/3.],[2,.5,3,1/3.],[2,.5],[2,.5]],
                             anchor_steps=[8,16,32,64,100,300],anchor_offset=0.5,normalizations=[20,-1,-1,-1,-1,-1],prior_scaling=[0.1,0.1,0.2,0.2])
    def __init__(self,params=None):
        if isinstance(params,SSDParams):
            self.params=params
        else:
            self.params=SSDNet.default_params

    def net(self,inputs,is_training=True,update_feat_shapes=True,dropout_keep_prob=0.5,prediction_fn=slim.softmax,reuse=None,scope="ssd_300_vgg"):
        r=ssd_net(inputs,num_classes=self.params.num_classes,
                  feat_layers=self.params.feat_layers,
                  anchor_sizes=self.params.anchor_sizes,
                  anchor_ratios=self.params.anchor_ratios,
                  normalizations=self.params.normalizations,
                  is_training=is_training,
                  dropout_keep_prob=dropout_keep_prob,
                  prediction_fn=prediction_fn,
                  reuse=reuse,
                  scope=scope)
        if update_feat_shapes:
            shapes=ssd_feat_shapes_from_net(r[0],self.params.feat_shapes)
            self.params=self.params._replace(feat_shapes=shapes)
        return r

    def arg_scope(self,weight_decay=0.0005,data_format="NHWC"):
        return ssd_arg_scope(weight_decay,data_format=data_format)

    def updata_feature_shapes(self,predictions):
        shapes=ssd_feat_shapes_from_net(predictions,self.params.feat_shapes)
        self.params=self.params._replace(feat_shapes=shapes)

    def anchors(self,img_shape,dtype=np.float32):
        return ssd_anchors_all_layers(img_shape,self.params.feat_shapes,self.params.anchor_sizes,self.params.anchor_ratios,
                                      self.params.anchor_steps,self.params.anchor_offset,dtype)

    def bboxes_encode(self,labels,bboxes,anchors,scope=None):
        return tf_ssd_bboxes_encode(labels,bboxes,anchors,self.params.num_classes,self.params.no_annotaion_label,
                                    ignore_threshold=0.5,prior_scaling=self.params.prior_scaling,scope=scope)

    def bboxes_decode(self,feat_localizations,anchors,scope="ssd_bboxes_decode"):
        return tf_ssd_bboxes_decode(feat_localizations,anchors,prior_scaling=self.params.prior_scaling,scope=scope)

    def detected_bboxes(self,predictions,localizations,select_threshold=None,nms_threshold=0.5,
                        clipping_bbox=None,top_k=400,keep_top_k=200):
        rscores,rbboxes=tf_ssd_bboxes_select([predictions,localizations],select_threshold=select_threshold,num_classes=self.params.num_classes)
        rscores,rbboxes=tfe.bboxes_sort(rscores,rbboxes,top_k=top_k)
        rscores,rbboxes=tfe.bboxes_num_batch(rscores,rbboxes,num_threshold=nms_threshold,keep_top_k=keep_top_k)
        if clipping_bbox is None:
            rbboxes=tfe.bboxes_clip(clipping_bbox,rbboxes)
        return rscores,rbboxes

    def losses(self,logits,localisations,gclasses,glocalisations,gscores,match_threshold=0.5,negative_ratio=3.,alpha=1.,label_smoothing=0.,scope="ssd_losses"):
        return ssd_losses(logits,localisations,gclasses,glocalisations,gscores,match_threshold=match_threshold,negative_ratio=negative_ratio,
                          alpha=alpha,label_smoothing=label_smoothing,scope=scope)

def ssd_size_bounds_to_values(size_bounds,n_feat_layers,img_shape=(300,300)):
    assert img_shape[0]==img_shape[1]
    img_size=img_shape[0]
    min_ratio=int(size_bounds[0]*100)
    max_ratio=int(size_bounds[1]*100)
    step=int(math.floor(max_ratio-min_ratio)/(n_feat_layers-2))
    sizes=[[img_size*size_bounds[0]/2,img_size*size_bounds[0]]]
    for ratio in range(min_ratio,max_ratio+1,step):
        sizes.append((img_size*ratio/100.,
                      img_size*(ratio+step)/100.))
    return sizes

def ssd_feat_shapes_from_net(predictions,default_shapes=None):
    feat_shapes=[]
    for l in predictions:
        if isinstance(l,np.ndarray):
            shape=l.shape
        else:
            shape=l.get_shape().as_list()
        shape=shape[1:4]
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes

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
        w[i+di]=sizes[0]/img_shape[1]/math.sqrt(r)
    return y,x,h,w

def ssd_anchors_all_layers(img_shape,layers_shape,anchors_sizes,anchor_ratios,
                           anchor_steps,offset=0.5,dtype=np.float32):
    layers_anchors=[]
    for i,s in enumerate(layers_shape):
        anchor_bboxes=ssd_anchor_one_layer(img_shape,s,anchor_sizes[i],anchor_ratios[i],anchor_steps[i],offset=offset,dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors

def tensor_shape(x,rank=3):
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape=x.get_shape().with_rank(rank).as_list()
        dynamic_shape=tf.unstack(tf.shape(x),rank)
        return [s if s is not None else d for s,d in zip(static_shape,dynamic_shape)]

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
    cls_pred=slim.conv2d(net,num_cls_pred,[3,3],activation_fn=None,scope="conv_cls")
    cls_pred=channel_to_last(cls_pred)
    cls_pred=tf.reshape(cls_pred,tensor_shape(cls_pred,4)[:-1]+[num_anchors,num_classes])
    return cls_pred,loc_pred

def ssd_net(inputs,num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,prediction_fn=slim.softmax,reuse=None,scope="ssd_300_vgg"):
    end_points={}
    with tf.variable_scope(scope,"ssd_300_vgg",[inputs],reuse=reuse):
        net=slim.peat(inputs,2,slim.conv2d,64,[3,3],scope="conv1")
        end_points["block1"]=net
        net=slim.max_pool2d(net,[2,2],scope="pool1")

        net=slim.repeat(net,2,slim.conv2d,128,[3,3],scope="conv2")
        end_points["block2"]=net
        net=slim.max_pool2d(net,[2,2],scope="pool2")

        net=slim.repeat(net,3,slim.conv2d,256,[3,3],scope="conv3")
        end_points["block3"]=net
        net=slim.max_pool2d(net,[2,2],scope="pool3")

        net=slim.repeat(net,3,3,slim.conv3d,512,[3,3],scope="conv4")
        end_points["block4"]=net
        net=slim.max_pool2d(net,[2,2],scope="pool4")

        net=slim.repeat(net,3,slim.conv2d,512,[3,3],scope="conv5")
        end_points["block5"]=net
        net=slim.max_pool2d(net,[3,3],stride=1,scope="pool5")

        net=slim.conv2d(net,1024,[3,3],rate=6,scope="conv6")
        end_points["block6"]=net
        net=tf.layers.dropout(net,rate=dropout_keep_prob,training=is_training)

        net=slim.conv2d(net,1024,[1,1],scope="conv7")
        end_points["block7"]=net
        net=tf.layers.dropout(net,rate=dropout_keep_prob,training=is_training)

        end_point="block8"
        with tf.variable_scope(end_point):
            net=slim.conv2d(net,128,[1,1],scope="conv1*1")
            net=pad2d(net,pad=(1,1))
            net=slim.conv2d(net,256,[3,3],stride=2,scope="conv3*3",padding="VALID")

        end_points[end_point]=net
        end_point="block9"
        with tf.variable_scope(end_point):
            net=slim.conv2d(net,128,[1,1],scope="conv1*1")
            net=pad2d(net,pad=(1,1))
            net=slim.conv2d(net,256,[3,3],stride=2,scope="conv3*3",padding="VALID")

        end_points[end_point]=net
        end_point="block10"
        with tf.variable_scope(end_point):
            net=slim.conv2d(net,128,(1,1),scope="conv1*1")
            net=slim.conv2d(net,256,[3,3],stride=2,scope="conv3*3",padding="VALID")

        end_points[end_point]=net
        end_point="block11"
        with tf.variable_scope(end_point):
            net=slim.conv2d(net,128,[1,1],scope="conv1*1")
            net=slim.conv2d(net,256,[3,3],scope="conv3*3",padding="VALID")
        end_points[end_point]=net

        predictions=[]
        logits=[]
        localisations=[]
        for i,layer in enumerate(feat_layers):
            with tf.variable_scope(layer+"_box"):
                p,l=ssd_multibox_layer(end_points[layer],num_classes,anchor_sizes[i],anchor_ratios[i],normalizations[i])
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)

        return predictions,localisations,logits,end_points

ssd_net.default_image_size=300

def ssd_arg_scope(weight_decay=0.0005,data_format="NHWC"):
    with slim.arg_scope([slim.conv2d,slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d],padding="SAME",data_format=data_format):
            with slim.arg_scope([pad2d,l2_normalization,channel_to_last],data_format=data_format) as sc:
                return sc


def ssd_losses(logits,localisations,gclasses,glocalisations,gscores,match_threshold=0.5,negative_ratio=3.,alpha=1.,
               label_smoothing=0.,device="/cpu:0",scope=None):
    with tf.name_scope(scope,"ssd_losses"):
        lshape=tfe.get_shape(logits[0],5)
        num_classes=lshape[-1]
        batch_size=lshape[0]

        flogits=[]
        fgclasses=[]
        fgscores=[]
        flocalisations=[]
        fglocalisations=[]

        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i],[-1,num_classes]))
            fgclasses.append(tf.reshape(gclasses[i],[-1]))
            fgscores.append(tf.reshape(gclasses,[-1]))
            flocalisations.append(tf.reshape(localisations[i],[-1,4]))
            fglocalisations.append(tf.reshape(glocalisations[i],[-1,4]))

        logits=tf.concat(flogits,axis=0)
        gclasses=tf.concat(fgclasses,axis=0)
        gscores=tf.concat(fgscores,axis=0)
        localisations=tf.concat(flocalisations,axis=0)
        glocalisations=tf.concat(fglocalisations,axis=0)
        dtype=logits.dtype

        pmask=gscores>match_threshold
        fpmask=tf.cast(pmask,dtype)
        n_positives=tf.reduce_sum(fpmask)

        no_classes=tf.cast(pmask,tf.int32)
        predictions=slim.softmax(logits)
        nmask=tf.logical_and(tf.logical_not(pmask),gscores>-0.5)
        fnmask=tf.cast(nmask,dtype)
        nvalues=tf.where(nmask,predictions[:,0],1.-fnmask)
        nvalues_flat=tf.reshape(nvalues,[-1])
        max_neg_entries=tf.cast(tf.reduce_sum(fnmask),tf.int32)
        n_neg=tf.cast(negative_ratio*n_positives,tf.int32)+batch_size
        n_neg=tf.minimum(n_neg,max_neg_entries)

        val,idxes=tf.nn.top_k(-nvalues_flat,k=n_neg)
        max_hard_pred=-val[-1]
        nmask=tf.logical_and(nmask,nvalues<max_hard_pred)
        fnmask=tf.cast(nmask,dtype)

        with tf.name_scope("cross_entropy_pos"):
            loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=gclasses)
            loss=tf.div(tf.reduce_sum(loss*fpmask),batch_size,name="value")
            tf.losses.add_loss(loss)
        with tf.name_scope("cross_entropy_neg"):
            loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=no_classes)
            loss=tf.div(tf.reduce_sum(loss*fnmask),batch_size,name="value")
            tf.losses.add_loss(loss)
        with tf.name_scope("localisation"):
            weights=tf.expand_dims(alpha*pmask,axis=-1)
            loss=abs_smooth(localisations-glocalisations)
            loss=tf.div(tf.reduce_sum(loss*weights),batch_size,name="value")
            tf.losses.add_loss(loss)

    def ssd_losses_old(logits,localisations,gclasses,glocalisations,gscores,match_threshold=0.5,negative_ratio=3.,alpha=1.,label_smoothing=0.,device="/cpu:0",scope=None):
        with tf.device(device):
            with tf.name_scope(scope,"ssd_losses"):
                l_cross_pos=[]
                l_cross_neg=[]
                l_loc=[]

                for i in range(len(logits)):
                    with tf.name_scope("block_%i"%i):
                        wsize=tfe.get_shape(logits[i],rank=5)
                        wsize=wsize[1]*wsize[2]*wsize[3]

                        pmask=gscores[i]>match_threshold
                        fpmask=tf.cast(pmask,dtype)
                        n_positives=tf.reduce_sum(fpmask)

                        no_classes=tf.cast(pmask,tf.int32)
                        predictions=slim.softmax(logits[i])
                        nmask=tf.logical_and(tf.logical_not(pmask),gscores[i]>-0.5)
                        fnmask=tf.cast(nmask,dtype)
                        nvalues=tf.where(nmask,predictions[:,:,:,:,0],1.-fnmask)
                        nvalues_flat=tf.reshape(nvalues,[-1])

                        n_neg=tf.cast(negative_ratio*n_positives,tf.int32)
                        n_neg=tf.maximum(n_neg,tf.size(nvalues_flat)//8)
                        n_neg=tf.maximum(n_neg,tf.shape(nvalues)[0]*4)
                        max_neg_entries=1+tf.cast(tf.reduce_sum(fnmask),tf.int32)
                        n_neg=tf.minimum(n_neg,max_neg_entries)

                        val,idxes=tf.nn.top_k(-nvalues_flat,k=n_neg)
                        max_hard_pred=-val[-1]
                        nmask=tf.logical_and(nmask,nvalues<max_hard_pred)
                        fnmask=tf.cast(nmask,dtype)

                        with tf.name_scope("cross_entropy_pos"):
                            fpmask=wsize*fpmask
                            loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],labels=gclasses[i])
                            loss=tf.losses.computed_weighted_loss(loss,fpmask)
                            l_cross_pos.append(loss)

                        with tf.name_scope("cross_entropy_neg"):
                            fnmask=wsize*fnmask
                            loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],labels=no_classes)
                            loss=tf.losses.computed_weighted_loss(loss,fnmask)
                            l_cross_neg.append(loss)

                        with tf.name_scope("localization"):
                            weights=tf.expand_dims(alpha*fpmask,axis=-1)
                            loss=abs_smooth(localisations[i]-glocalisations[i])
                            loss=tf.losses.computed_weighted_loss(loss,weights)
                            l_loc.append(loss)

                    with tf.name_scope("total"):
                        total_cross_pos=tf.add_n(l_cross_pos,"cross_entropy_pos")
                        total_cross_neg=tf.add_n(l_cross_neg,"cross_entropy_neg")
                        total_cross=tf.add(total_cross_pos,total_cross_neg,"cross_entropy")
                        total_loc=tf.add_n(l_loc,"localization")

                        tf.add_to_collections("EXTRA_LOSSES",total_cross_pos)
                        tf.add_to_collections("EXTRA_LOSSES",total_cross_neg)
                        tf.add_to_collections("EXTRA_LOSSES",total_cross)
                        tf.add_to_collections("EXTRA_LOSSES",total_loc)









        

