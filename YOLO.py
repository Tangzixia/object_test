#coding=utf-8

import tensorflow as tf
import cv2

def leak_relu(x,alpha):
    return tf.maximum(alpha*x,x)

class YOLO(object):
    def __init__(self,weights_file,verbose=True):
        
