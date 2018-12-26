#coding=utf-8


import numpy as np
import tensorflow as tf
import tf_extended as tfe
from custom_layers import *


def tf_ssd_bboxes_encode_layer(labels,bboxes,anchors_layer,num_classes,no_annotaion_label,ignore_threshold=0.5,prior_scaling=[0.1,0.1,0.2,0.2],dtype=tf.flota32):
    '''
    anchor_layer:代表
    :param labels:
    :param bboxes:
    :param anchor_layer:
    :param num_classes:
    :param no_annotaion_label:
    :param ignore_threshold:
    :param prior_scaling:
    :param dtype:
    :return:
    '''
    yref,xref,href,wref=anchors_layer
    # 这样结束后可以扩增维度，（N,M,1)--->(N,M,count)
    ymin=yref-href/2
    ymax=yref+href/2
    xmin=xref-wref/2
    xmax=xref+wref/2
    vol_anchors=(xmax-xmin)*(ymax-ymin)

    # shape为(feature_map_height, feature_map_width, anchors_per_feature_map_point)
    shape=(yref.shape[0],yref.shape[1],href.size)
    feat_labels=tf.zeros(shape,dtype=tf.int64)
    feat_scores=tf.zeros(shape,dtype=dtype)
    feat_ymin=tf.zeros(shape,dtype=dtype)
    feat_xmin=tf.zeros(shape,dtype=dtype)
    feat_ymax=tf.ones(shape,dtype=dtype)
    feat_xmax=tf.ones(shape,dtype=dtype)


    def jaccard_with_anchors(bbox):
        int_ymin=tf.maximum(ymin,bbox[0])
        int_xmin=tf.maximum(xmin,bbox[0])
        int_ymax=tf.minimum(ymax,bbox[2])
        int_xmax=tf.minimum(xmax,bbox[3])
        h=tf.maximum(int_ymax-int_ymin,0.)
        w=tf.maximum(int_xmax-int_xmin,0.)

        inter_vol=h*w
        union_vol=vol_anchors-inter_vol+(bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        jaccard=tf.div(inter_vol,union_vol)
        return jaccard

    def intersection_with_anchors(bbox):
        int_ymin=tf.maximum(ymin,bbox[0])
        int_xmin=tf.maximum(xmin,bbox[0])
        int_ymax=tf.minimum(ymax,bbox[2])
        int_xmax=tf.minimum(xmax,bbox[3])

        h=tf.maximum(int_ymax-int_ymin,0.)
        w=tf.maximum(int_xmax-int_xmin,0.)
        inter_vol=h*w
        scores=tf.div(inter_vol,vol_anchors)
        return scores

    def condition(i,feat_labels,feat_scores,feat_ymin,feat_ymax,feat_xmin,feat_xmax):
        r=tf.less(i,tf.shape(labels))
        return r[0]

    def body(i,feat_labels,feat_scores,feat_ymin,feat_ymax,feat_xmin,feat_xmax):
        label=labels[i]
        bbox=bboxes[i]
        jaccard=jaccard_with_anchors(bbox)

        # 判断条件如下：
        # cur_jaccard>scores && jaccard>jaccard_threshold && scores>-0.5
        # 先验框匹配规则：
        # 1、根据ground truth找先验框：找到与每个ground truth的IOU最大的候选矿（正样本数不足）；
        # 2、根据先验框找对应的ground truth：从剩余的候选框中找与ground truth大于thresh的候选框，设置为正样本；
        mask=tf.greater(jaccard,feat_scores)
        mask=tf.logical_and(mask,feat_scores>-0.5)
        mask=tf.logical_and(mask,label<num_classes)
        imask=tf.cast(mask,tf.int64)
        fmask=tf.cast(mask,dtype)

        feat_labels=imask*label+(1-imask)*feat_labels
        feat_scores=tf.where(mask,jaccard,feat_scores)
        feat_ymin=fmask*bbox[0]+(1-fmask)*feat_ymin
        feat_xmin=fmask*bbox[1]+(1-fmask)*feat_xmin
        feat_ymax=fmask*bbox[2]+(1-fmask)*feat_ymax
        feat_xmax=fmask*bbox[3]+(1-fmask)*feat_xmax

        return [i+1,feat_labels,feat_scores,feat_ymin,feat_ymax,feat_xmin,feat_xmax]

    i=0
    [i,feat_labels,feat_scores,feat_ymin,feat_ymax,
     feat_xmin,feat_xmax]=tf.while_loop(condition,body,[i,feat_labels,feat_scores,feat_ymin,feat_ymax,feat_xmin,feat_xmax])

    feat_cy=(feat_ymax+feat_ymin)/2.
    feat_cx=(feat_xmax+feat_xmin)/2.
    feat_h=feat_ymax-feat_ymin
    feat_w=feat_xmax-feat_xmin

    feat_cy=(feat_cy-yref)/href/prior_scaling[0]
    feat_cx=(feat_cx-xref)/wref/prior_scaling[1]
    feat_h=tf.log(feat_h/href)/prior_scaling[2]
    feat_w=tf.log(feat_w/wref)/prior_scaling[3]

    feat_localizations=tf.stack([feat_cx,feat_cy,feat_w,feat_h],axis=-1)
    return feat_labels,feat_localizations,feat_scores

def tf_ssd_bboxes_encode(labels,bboxes,anchors,matching_threshold=0.5,prior_scaling=[0.1,0.1,0.2,0.2],dtype=tf.float32,scope="ssd_bboxes_encode"):

    with tf.name_scope(scope):
        target_labels=[]
        target_localizations=[]
        target_scores=[]
        for i,anchors_layer in enumerate(anchors):
            with tf.name_scope("bboxes_encode_block_%i"%i):
                t_labels,t_loc,t_scores=tf_ssd_bboxes_encode_layer(labels,bboxes,anchors_layer,matching_threshold,prior_scaling,dtype)
                target_labels.append(t_labels)
                target_localizations.append(t_loc)
                target_scores.append(t_scores)

        return target_labels,target_localizations,target_scores


def tf_ssd_bboxes_decode_layer(feat_locations,anchor_layer,prior_scaling=[0.1,0.1,0.2,0.2]):
    yref,xref,href,wref=anchor_layer
    cx=feat_locations[:,:,:,0]*wref*prior_scaling[0]+xref
    cy=feat_locations[:,:,:,1]*href*prior_scaling[1]+yref
    w=wref*tf.exp(feat_locations[:,:,:,2]*prior_scaling[2])
    h=href*tf.exp(feat_locations[:,:,:,3]*prior_scaling[3])

    ymin=cy-h/2.
    xmin=cx-w/2.
    ymax=cy+h/2.
    xmax=cx+w/2.

    bboxes=tf.stack([ymin,xmin,ymax,xmax],axis=-1)
    return bboxes

def tf_ssd_bboxes_decode(feat_localizations,anchors,prior_scaling=[0.1,0.1,0.2,0.2],scope="ssd_bboxes_decode"):
    with tf.name_scope(scope):
        bboxes=[]
        for i,anchor_layer in enumerate(anchors):
            bboxes.append(tf_ssd_bboxes_decode_layer(feat_localizations[i],anchor_layer,prior_scaling))
        return bboxes


def tf_ssd_bboxes_select_layer(predictions_layer,localizations_layer,select_threshold=None,num_classes,ignore_class=0,scope=None):
    select_threshold=0.0 if select_threshold is None else select_threshold
    with tf.name_scope(scope,"ssd_bboxes_select_layer",[predictions_layer,localizations_layer]):
        p_shape=tfe.get_shape(predictions_layer)
        predictions_layer=tf.reshape(predictions_layer,tf.stack([p_shape[0],-1,p_shape[-1]]))

        l_shape=tfe.get_shape(localizations_layer)
        localizations_layer=tf.reshape(localizations_layer,tf.stack([l_shape[0],-1,l_shape[-1]]))

        d_scores={}
        d_bboxes={}
        for c in range(0,num_classes):
            if c!=ignore_class:
                scores=predictions_layer[:,:,c]
                fmask=tf.cast(tf.greater_equal(scores,select_threshold),scores.dtype)
                scores=scores*fmask
                bboxes=localizations_layer*tf.expand_dims(fmask,axis=-1)

                d_scores[c]=scores
                d_bboxes[c]=bboxes
        return d_scores,d_bboxes

def tf_ssd_bboxes_select(predictions_set,localizations_set,select_threshold=None,num_classes=21,ignore_class=0,scope=None):
    with tf.name_scope(scope,"ssd_bboxes_select",[predictions_set,localizations_set]):
        l_scores=[]
        l_bboxes=[]
        for i in range(len(predictions_set)):
            scores,bboxes=tf_ssd_bboxes_select_layer(predictions_set[i],localizations_set[i],select_threshold,num_classes,ignore_class)

            l_scores.append(scores)
            l_bboxes.append(bboxes)
        d_scores={}
        d_bboxes={}
        for c in l_scores[0].keys():
            ls=[s[c] for s in l_scores]
            lb=[b[c] for b in l_bboxes]
            d_scores[c]=tf.concat(ls,axis=1)
            d_bboxes[c]=tf.concat(lb,axis=1)
        return d_scores,d_bboxes

def tf_ssd_bboxes_select_layer_all_classes(predictions_layer,localizations_layer,select_threshold=None):
    p_shape=tfe.get_shape(predictions_layer)
    predictions_layer=tf.reshape(predictions_layer,tf.stack([p_shape[0],-1,p_shape[-1]]))
    l_shape=tfe.get_shape(localizations_layer)
    localizations_layer=tf.reshape(localizations_layer,tf.stack([l_shape[0],-1,l_shape[-1]]))

    if select_threshold is None or select_threshold==0:
        classes=tf.argmax(predictions_layer,axis=2)
        scores=tf.reduce_max(predictions_layer,axis=2)
        scores=scores*tf.cast(classes>0,scores.dtype)
    else:
        sub_predictions=predictions_layer[:,:,1:]
        classes=tf.argmax(sub_predictions,axis=2)+1
        scores=tf.reduce_max(sub_predictions,axis=2)
        mask=tf.greater(scores,select_threshold)
        classes=classes*tf.cast(mask,classes.dtype)
        scores=scores*tf.cast(mask,scores.dtype)
    bboxes=localizations_layer
    return classes,scores,bboxes

def tf_ssd_bboxes_select_all_classes(predictions_net,localizations_net,select_threshold=None,scope=None):
    with tf.name_scope(scope,"ssd_bboxes_select",[predictions_net,localizations_net]):
        l_classes=[]
        l_scores=[]
        l_bboxes=[]
        for i in range(len(predictions_net)):
            classes,scores,bboxes=tf_ssd_bboxes_select_layer_all_classes(predictions_net[i],localizations_net[i],select_threshold)

        classes=tf.concat(l_classes,axis=1)
        scores=tf.concat(l_scores,axis=1)
        bboxes=tf.concat(l_bboxes,axis=1)
        return classes,scores,bboxes

def ssd_losses(logits,localisations,gclasses,glocalisations,gscores,match_threshold=0.5,negative_ratio=3.,alpha=1.,label_smoothing=0.,device="/cpu:0",scope=None):
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
            fgscores.append(tf.reshape(gscores[i],[-1]))
            flocalisations.append(tf.reshape(localisations[i],[-1,4]))
            fglocalisations.append(tf.reshape(glocalisations[i],[-1,4]))

        logits=tf.concat(flogits,axis=0)
        gclasses=tf.concat(gclasses,axis=0)
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
            # tf.losses.add_loss(loss)
            tf.add_to_collection("total_loss",loss)

        with tf.name_scope("localization"):
            weights=tf.expand_dims(alpha*fpmask,axis=-1)
            loss=abs_smooth(localisations-glocalisations)
            loss=tf.div(tf.reduce_sum(loss*weights),batch_size,name="value")
            # tf.losses.add_loss(loss)
            tf.add_to_collection("total_loss",loss)


def ssd_losses_old(logits,localisations,gclasses,glocalisations,gscores,
                   match_threshold=0.5,negative_ratio=3.,alpha=1.,label_smoothing=0.,device="/cpu:0",scope=None):
    with tf.device(device):
        with tf.name_scope(scope,"ssd_losses"):
            l_cross_pos=[]
            l_cross_neg=[]
            l_loc=[]
            for i in range(len(logits)):
                dtype=logits[i].dtype
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
                    # 最后的负样本
                    nmask=tf.logical_and(nmask,nvalues<max_hard_pred)
                    fnmask=tf.cast(nmask,dtype)

                    # cross_entropy_loss
                    with tf.name_scope("cross_entropy_pos"):
                        fpmask=wsize*fpmask
                        loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],labels=gclasses[i])
                        loss=tf.losses.compute_weighted_loss(loss,fpmask)
                        l_cross_pos.append(loss)

                    with tf.name_scope("cross_entropy_neg"):
                        fnmask=wsize*fnmask
                        loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=no_classes)
                        loss=tf.losses.compute_weighted_loss(loss,fnmask)
                        l_cross_neg.append(loss)
                    # 定位损失
                    with tf.name_scope("localization"):
                        weights=tf.expand_dims(alpha*fpmask,axis=-1)
                        loss=custome_layers.abs_smooth(localisations[i]-glocalisations[i])
                        loss=tf.losses.compute_weighted_loss(loss,weight)
                        l_loc.append(loss)

                # 添加所有的损失
                with tf.name_scope("total_loss"):
                    total_cross_pos=tf.add_n(l_cross_pos,"cross_entropy_pos")
                    total_cross_neg=tf.add_n(l_cross_neg,"cross_entropy_neg")
                    total_cross=tf.add(total_cross_pos,total_cross_neg,"cross_entropy")
                    total_loc=tf.add_n(l_loc,"localization")

                    tf.add_to_collection("EXTRA_LOSSES",total_cross_pos)
                    tf.add_to_collection("EXTRA_LOSSES",total_cross_neg)
                    tf.add_to_collection("EXTRA_LOSSES",total_cross)
                    tf.add_to_collection("EXTRA_LOSSES",total_loc)
