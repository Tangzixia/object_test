#coding=utf-8

from keras.backend as k
import tensorflow as tf
from config import ignore_thresh
from detect_function import yolo_head

def compute_loss(yolo_outputs,y_true,anchors,num_classes,ignore_thresh=ignore_thresh,print_loss=False):
    anchor_mask=[[6,7,8],[3,4,5],[0,1,2]]

    input_shape=k.cast(k.shape(yolo_outputs[0])[1:3]*32,k.dtype(y_true[0]))
    grid_shapes=[k.cast(k.shape(yolo_outputs[1])[1:3],k.dtype(y_true[0])) for l in range(3)]
    loss=0
    m=k.shape(yolo_outputs[0])[0]
    mf=k.cast(m,k.dtype(yolo_outputs[0]))

    for l in range(3):
        object_mask=y_ture[l][...,4:5]
        grid,raw_pred,pred_xy,pred_wh=yolo_head(yolo_outputs[l],anchors[anchor_mask[l]],num_classes,input_shape,calc_loss=True)

        pred_box=k.concatnate([pred_xy,pred_wh])

        raw_true_xy=y_true[l][...,:2]*grid_shapes[l][::-1]-grid
        raw_true_wh=k.log(y_true[l][...,2:4]/anchors[anchor_mask[l]]*input_shape[::-1])
        raw_true_wh=k.switch(object_mask,raw_true_wh,k.zeros_like(raw_true_wh))
        box_loss_scale=2-y_true[l][...,2:3]*y_true[l][...,3:4]

        ignore_mask=tf.TensorArray(k.dtype(y_true[0]),size=1,dynamic_size=True)
        object_mask_bool=k.cast(object_mask,"bool")

        def loop_body(b,ignore_mask):
            true_box=tf.boolean_mask(y_true[l][b,...,0:4],object_mask_bool[b,...,0])
            iou=box_Iou(pred_box[b],true_box)
            best_iou=k.max(iou,axis=-1)
            ignore_mask=ignore_mask.write(b,k.cast(best_iou<ignore_thresh,k.dtype(ture_box)))
            return b+1,ignore_mask

        _,ignore_mask=k.control_flow_ops.while_loop(lambda b,*args:b<m,loop_body,[0,ignore_mask])
        ignore_mask=ignore_mask.stack()
        ignore_mask=k.expand_dims(ignore_mask,-1)

        xy_loss=object_mask*box_loss_scale*k.binary_crossentropy(raw_true_xy,raw_pred[...,0:2],from_logits=True)
        wh_loss=object_mask*box_loss_scale*0.5*k.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss=object_mask*k.binary_crossentropy(object_mask,raw_pred[...,4:5],from_logits=True)+(1-object_mask)*k.binary_crossentropy(object_mask,raw_pred[...,4:5],from_logits=True)*ignore_mask

        class_loss=object_mask*k.binary_crossentropy(true_class_probs,raw_pred[...,5:],from_logits=True)

        xy_loss=k.sum(xy_loss)/mf
        wh_loss=k.sum(wh_loss)/mf
        confidence_loss=k.sum(confidence_loss)/mf
        class_loss=k.sum(class_loss)/mf
        loss+=xy_loss+wh_loss+confidence_loss+class_loss

        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                            message='loss: ')
    return loss

def box_IoU(b1, b2):
    """
    Calculer IoU between 2 BBs
    # hoi bi nguoc han tinh left bottom, right top TODO
    :param b1: predicted box, shape=[None, 13, 13, 3, 4], 4: xywh
    :param b2: true box, shape=[None, 13, 13, 3, 4], 4: xywh
    :return: iou: intersection of 2 BBs, tensor, shape=[None, 13, 13, 3, 1] ,1: IoU
    b = tf.cast(b, dtype=tf.float32)
    """
    with tf.name_scope('BB1'):
        """Calculate 2 corners: {left bottom, right top} and area of this box"""
        b1 = tf.expand_dims(b1, -2)  # shape= (None, 13, 13, 3, 1, 4)
        b1_xy = b1[..., :2]  # x,y shape=(None, 13, 13, 3, 1, 2)
        b1_wh = b1[..., 2:4]  # w,h shape=(None, 13, 13, 3, 1, 2)
        b1_wh_half = b1_wh / 2.  # w/2, h/2 shape= (None, 13, 13, 3, 1, 2)
        b1_mins = b1_xy - b1_wh_half  # x,y: left bottom corner of BB
        b1_maxes = b1_xy + b1_wh_half  # x,y: right top corner of BB
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]  # w1 * h1 (None, 13, 13, 3, 1)

    with tf.name_scope('BB2'):
        """Calculate 2 corners: {left bottom, right top} and area of this box"""
        # b2 = tf.expand_dims(b2, -2)  # shape= (None, 13, 13, 3, 1, 4)
        b2 = tf.expand_dims(b2, 0)  # shape= (1, None, 13, 13, 3, 4)  # TODO 0?
        b2_xy = b2[..., :2]  # x,y shape=(None, 13, 13, 3, 1, 2)
        b2_wh = b2[..., 2:4]  # w,h shape=(None, 13, 13, 3, 1, 2)
        b2_wh_half = b2_wh / 2.  # w/2, h/2 shape=(None, 13, 13, 3, 1, 2)
        b2_mins = b2_xy - b2_wh_half  # x,y: left bottom corner of BB
        b2_maxes = b2_xy + b2_wh_half  # x,y: right top corner of BB
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]  # w2 * h2

    with tf.name_scope('Intersection'):
        """Calculate 2 corners: {left bottom, right top} based on BB1, BB2 and area of this box"""
        # intersect_mins = tf.maximum(b1_mins, b2_mins, name='left_bottom')  # (None, 13, 13, 3, 1, 2)
        intersect_mins = K.maximum(b1_mins, b2_mins)  # (None, 13, 13, 3, 1, 2)
        # intersect_maxes = tf.minimum(b1_maxes, b2_maxes, name='right_top')  #
        intersect_maxes = K.minimum(b1_maxes, b2_maxes)
        # intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)  # (None, 13, 13, 3, 1, 2), 2: w,h
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # intersection: wi * hi (None, 13, 13, 3, 1)

    IoU = tf.divide(intersect_area, (b1_area + b2_area - intersect_area), name='divise-IoU')  # (None, 13, 13, 3, 1)

    return IoU