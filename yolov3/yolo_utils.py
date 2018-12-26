#coding=utf-8

import random
import cv2
import numpy as np
from PIL import Image
import os

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names=f.readlines()
    class_names=[c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors=f.readline()
        anchors=[float(x) for x in anchors.split(",")]
        anchors=np.array(anchors).reshape(-1,2)
    return anchors

def get_training_data(annotation_path,data_path,input_shape,anchors,num_classes,max_boxes=100,load_previous=True):
    if load_previous==True and os.path.isfile(data_path):
        data=np.load(data_path)
        print("Loading training data from "+data_path)
        return data["image_data"],data["box_data"],data["image_shape"],[data["y_true0"],data["y_true1"],data["y_true2"]]
    image_data=[]
    box_data=[]
    image_shape=[]
    with open(annotation_path) as f:
        GG=f.readlines()
        np.random.shuffle(GG)
        for line in (GG):
            line=line.split(" ")
            filename=line[0]
            if filename[-1]=="\n":
                filename=filename[:-1]
            image=Image.open(filename)
            boxed_image,shape_image=letterbox_image(image,tuple(reversed(input_shape)))
            image_data.append(np.array(boxed_image,dtype=np.uint8))
            image_shape.append(np.array(shape_image))

            boxes=np.zeros((max_boxes,5),dtype=np.int32)
            if(len(line)==1):
                box_data.append(boxes)
            for i,box in enumerate(line[1:]):
                if i<max_boxes:
                    boxes[i]=np.arary(list(map(int,box.split(","))))
                else:
                    break

            image_size=np.array(image.size)
            input_size=np.array(input_shape[::-1])
            new_size=(image_size*np.min(input_size/image_size)).astype(np.int32)
            boxes[i:i+1,0:2]=(boxes[i:i+1,0:2]*new_size/image_size+(input_size-new_size)/2).astype(np.int32)
            boxes[i,i+1,2:4]=(boxes[i:i+1,2:4]*new_size/image_size+(input_size-new_size)/2).astype(np.int32)
        box_data.append(boxes)

    image_shape=np.array(image_shape)
    image_data=np.array(image_data)
    box_data=(np.array(box_data))
    y_true=preprocess_true_boxes(box_data,input_shape[0],anchors,num_classes)
    np.savez(data_path,image_data=image_data,box_data=box_data,image_shape=image_shape,y_true0=y_true[0],y_true1=y_true[1],y_true2=y_true[2])
    print("Saving training data into "+data_path)
    return image_data,box_data,image_shape,y_true

def letterbox_image(image,size):
    image_w,image_h=image.size
    image_shape=np.array([image_h,image_w])
    w,h=size
    new_w=int(image_w*min(w/image_w,h/image_h))
    new_h=int(image_h*min(w/image_w,h/image_h))
    resized_image=image.resize((new_w,new_h),Image.BICUBIC)

    boxed_image=Image.new("RGB",size,(128,128,128))
    boxed_image.paste(resized_image,((w-new_w)//2,(h-new_h)//2))
    return boxed_image,image_shape

def preprocess_true_boxes(true_boxes,Input_shape,anchors,num_classes):
    assert (true_boxes[...,4]<num_classes).all(),"class id must be less than num_classes"
    anchor_mask=[[6,7,8],[3,4,5],[0,1,2]]
    true_boxes=np.array(true_boxes,dtype=np.float32)
    input_shape=np.array([Input_shape,Input_shape],dtype=np.int32)
    boxes_xy=(true_boxes[...,0:2]+true_boxes[...,2:4])//2
    boxes_wh=(true_boxes[...,2:4]-true_boxes[...,0:2])

    true_boxes[...,0:2]=boxes_xy/input_shape[::-1]
    true_boxes[...,2:4]=boxes_wh/input_shape[::-1]

    N=true_boxes.shape[0]
    grid_shapes=(input_shape//{0:32,1:16,2:8}[l] for l in range(3))

    y_true=[np.zeros((N,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+int(num_classes)),
                    dtype=np.float32) for l in range(3)]

    anchors=np.expand_dims(anchors,0)
    anchor_maxes=anchors/2.
    anchor_mins=-anchor_maxes
    valid_mask=boxes_wh[...,0]>0

    for b in (range(N)):
        wh=boxes_wh[b,valid_mask[b]]
        if len(wh)==0:
            continue
        wh=np.expand_dims(wh,-2)
        box_maxes=wh/2.
        box_mins=-box_maxes

        intersect_mins=np.maximum(box_mins,anchor_mins)
        intersect_maxes=np.minimum(box_maxes,anchor_maxes)
        intersect_wh=np.maximum(intersect_maxes-intersect_mins,0.)
        intersect_area=intersect_wh[...,0]*intersect_wh[...,1]
        box_area=wh[...,0]*wh[...,1]
        anchor_area=anchors[...,0]*anchors[...,1]
        iou=intersect_area/(box_area+anchor_area-intersect_area)
        best_anchor=np.argmax(iou,axis=-1)

        for t,n in enumerate(best_anchor):
            for l in range(3):
                if n in anchor_mask[l]:
                    i=np.floor(true_boxes[b,t,0]*grid_shapes[l][1].astype(np.int32))
                    j=np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype(np.int32)
                    if grid_shapes[l][1]==13 and (i>=13 or j>=13):
                        print(i)
                    k=anchor_mask[l].index(n)
                    c=true_boxes[b,t,4].astype(np.int32)
                    y_true[l][b,j,i,k,0:4]=true_boxes[b,t,0:4]
                    y_true[l][b,j,i,k,4]=1
                    y_true[l][b,j,i,k,5+c]=1
                    break
        return y_true


