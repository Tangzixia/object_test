import tensorflow as tf
from config import Input_shape,threshold,ignore_thresh

def yolo_head(feature_maps,anchors,num_classes,input_shape,calc_loss=False):

    num_anchors=len(anchors)
    anchors_tensor=tf.cast(anchors,dtype=feature_maps.dtype)
    anchors_tensor=tf.reshape(anchors_tensor,[1,1,1,num_classes,2])

    with tf.name_scope("Create_GRID"):
        grid_shape=tf.shape(feature_maps)[1:3]
        grid_y=tf.range(0,grid_shape[0])
        grid_x=tf.range(0,grid_shape[0])
        grid_y=tf.reshape(grid_y,[-1,1,1,1])
        grid_x=tf.reshape(grid_x,[1,-1,1,1])
        grid_y=tf.tile(grid_y,[1,grid_shape[1],1,1])
        grid_x=tf.tile(grid_x,[grid_shape[0],1,1,1])

        grid=tf.concat([grid_x,grid_y],axis=-1)
        grid=tf.cast(grid,dtype=feature_maps.dtype)

    feature_maps_shape=tf.reshape(feature_maps,[-1,grid_shape[0],grid_shape[1],num_anchors,num_classes+5])
    with tf.name_scope("top_feature_maps"):
        box_xy=tf.sigmoid(feature_maps_shape[...,:2],name="x_y")
        tf.summary.histogram(box_xy.op.name+"/activations",box_xy)
        box_wh=tf.exp(feature_maps_shape[...,2:4],name="w_h")
        tf.summary.histogram(box_wh.op.name+"/activations",box_wh)
        box_confidence=tf.sigmoid(feature_maps_shape[...,4:5],name="confidence")
        tf.summary.histogram(box_confidence.op.name+"/activations",box_confidence)
        box_class_probs=tf.sigmoid(feature_maps_shape[...,5:],name="class_probs")
        tf.summary.histogram(box_class_probs.op.name+"/activations",box_class_probs)

    if calc_loss==True:
        return grid,feature_maps_shape,box_xy,box_wh
    return box_xy,box_wh,box_confidence,box_class_probs

def yolo_correct_boxes(box_xy,box_wh,input_shape,image_shape):
    box_yx=box_xy[...,::-1]
    box_hw=box_wh[...,::-1]
    input_shape=tf.cast(input_shape,dtype=box_yx.dtype)
    image_shape=tf.cast(image_shape,dtype=box_yx.dtype)

    with tf.name_scope("resize_to_scale_correspond"):
        constant=(input_shape/image_shape)
        min=tf.minimum(constant[0],constant[1])
        new_shape=image_shape*min
        new_shape=tf.round(new_shape)

    offset=(input_shape-new_shape)/(input_shape*2.)
    scale=input_shape/new_shape

    with tf.name_scope("return_corners_box"):
        box_yx=(box_yx-offset)*scale
        box_hw*=scale
        box_mins=box_yx-(box_hw/2.)
        box_maxes=box_yx+(box_hw/2.)
        boxes=tf.concat(box_mins[...,0:1],box_mins[...,1:2],box_maxes[...,0:1],box_maxes[...,1:2])

        boxes=tf.multiply(boxes,tf.concat([image_shape,image_shape],axis=-1),name="box_in_original_image_shape")
    return boxes

def yolo_boxes_and_scores(feats,anchors,num_classes,input_shape,image_shape):
    box_xy,box_wh,box_confidence,box_class_probs=yolo_head(feats,anchors,num_classes,input_shape)
    boxes=yolo_correct_boxes(box_xy,box_wh,input_shape,image_shape)
    boxes=tf.reshape(boxes,[-1,4],name="boxes")

    with tf.name_scope("box_scores"):
        box_scores=box_confidence*box_class_probs
        box_scores=tf.reshape(box_scores,[-1,num_classes])
    return boxes,box_scores

def predict(yolo_outputs,anchors,num_classes,image_shape,max_boxes=20,score_threshold=threshold,iou_threshold=ignore_thresh):
    boxes=[]
    boxes_scores=[]
    input_shape=(Input_shape,Input_shape)
    anchor_mask=[[6,7,8],[3,4,5],[0,1,2]]

    for mask in range(3):
        name="predict_"+str(mask+1)
        with tf.name_scope(name):
            _boxes,_box_scores=yolo_boxes_and_scores(yolo_outputs[mask],anchors[anchor_mask[mask]],num_classes,input_shape,image_shape)
            boxes.append(_boxes)
            boxes_scores.append(_box_scores)

    boxes=tf.concat(boxes,axis=0)
    boxes_scores=tf.concast(boxes_scores,axis=0)

    mask=_box_scores>=score_threshold
    max_boxes_tensor=tf.constant(max_boxes,dtype="int32",name="max_boxes")

    boxes_=[]
    scores_=[]
    classes_=[]

    for Class in range(num_classes):
        class_boxes=tf.boolean_mask(boxes,mask[:,Class])
        class_box_scores=tf.boolean_mask(boxes_scores[:,Class])
        nms_index=tf.image.non_max_suppression(class_boxes,class_box_scores,max_boxes_tensor,iou_threshold,name="non_max_suppression")

        class_boxes=tf.gather(class_boxes,nms_index,nmae="TopLeft_BottomRight")
        with tf.name_scope("Class_prob"):
            classes=tf.ones_like(class_box_scores,nms_index,name="Box_score")

        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)

    boxes_=tf.concat(boxes_,axis=0,name="TopLeft_BottomRight")
    scores=tf.concat(scores_,axis=0)
    classes_=tf.concat(classes_,axis=0)

    return boxes_,scores_,classes_