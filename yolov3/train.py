#coding=utf-8

from config import Input_shape,channels,path
from network_function import YOLOv3
from loss_function import compute_loss
from yolo_utils import get_training_data,read_anchors,read_classes

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

import argparse
import numpy as np
import tensorflow as tf
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

np.random.seed(101)

def argument():
    parser=argparse.ArgumentParser(description="COCO or VOC or boat")
    parser.add_argument("--COCO",action="store_true",help="COCO flag")
    parser.add_argument("--VOC",actiion="store_true",help="VOC flag")
    parser.add_argument("--boat",action="store_true",help="boat flag")
    args=parser.parse_args()
    return args

PATH=path+"/yolov3"
classes_paths=PATH+"/model/boat_classes.txt"
classes_data=read_classes(classes_paths)
anchors_paths=PATH+"/model/yolo_anchors.txt"
anchors=read_anchors(anchors_paths)

annotation_path_train=PATH+"/model/boat_train.txt"
annotation_path_valid=PATH+"/model/boat_valid.txt"
annotation_path_test=PATH+"/model/boat_test.txt"

data_path_train=PATH+"/model/boat_train.npz"
data_path_valid=PATH+"/model/boat_valid.npz"
data_path_test=PATH+"/model/boat_test.npz"

VOC=False
args=argument()
if args.VOC==True:
    VOC=True
    classes_paths=PATH+"/model/voc_classes.txt"
    classes_data=read_classes(classes_paths)
    annotation_path_train=PATH+"/model/voc_train.txt"
    annotation_path_valid=PATH+"/model/voc_val.txt"
    data_path_train=PATH+"/model/voc_train.npz"
    data_path_valid=PATH+"/model/voc_valid.npz"


input_shape=(Input_shape,Input_shape)
x_train,box_data_train,image_shape_train,y_train=get_training_data()
x_valid,box_data_valid,image_shape_valid,y_valid=get_training_data()
x_test,box_data_test,image_shape_test,y_test=get_training_data()

number_image_train=np.shape(x_train)[0]
number_image_valid=np.shape(x_valid)[0]
number_image_test=np.shape(x_test)[0]

print("Starting 1st session...")
graph=tf.Graph()
with graph.as_default():
    global_step=tf.Variable(0,name="global_step",trainable=False)
    X=tf.placeholder(tf.float32,shape=[None,Input_shape,Input_shape,channels],name="Input")

    with tf.name_scope("Target"):
        Y1=tf.placeholder(tf.float32,shape=[None,Input_shape/32,Input_shape/32,3,(5+len(classes_data))],name="target_s1")
        Y2=tf.placeholder(tf.float32,shape=[None,Input_shape/16,Input_shape/16,3,(5+len(classes_data))],name="target_s2")
        Y3=tf.placeholder(tf.float32,shape=[None,Input_shape/8,Input_shape/8,3,(5+len(classes_data))],name="target_s3")

    x_reshape=tf.reshape(X,[-1,Input_shape,Input_shape,1])
    tf.summary.image("input",x_reshape)

    scale1,scale2,scale3=YOLOv3(x,len(classes_data)).feature_extractor()
    scale_total=[scale1,scale2,scale3]

    with tf.name_scope("Loss_and_Detect"):
        y_predict=[Y1,Y2,Y3]
        loss=compute_loss(scale_total,y_predict,anchors,len(classes_data),print_loss=False)
        tf.summary.scale("loss",loss)
    with tf.name_scope("Optimizer"):
        decay=0.0003
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss,global_step=global_step)

    saver=tf.train.Saver()
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config,graph=graph) as sess:
        summary_op=tf.summary.merge_all()
        if VOC==True:
            train_summary_writer=tf.summary.FileWriter(PATH+"/graphs_VOC1/train1",sess.graph)
            validation_summary_writer=tf.summary.FileWriter(PATH+"/graphs_VOC1/validation",sess.graph)
        else:
            train_summary_writer=tf.summary.FileWriter(PATH+"/graphs_boat10/train",sess.graph)
            validation_summary_writer=tf.summary.FileWriter(PATH+"/graphs_boat10/validation",sess.graph)

        init_op=tf.global_variables_initializer()
        sess.run(init_op)

        epochs=50
        batch_size=32
        best_loss_valid=10e6
        for epoch in range(epochs):
            start_time=time.time()
            mean_loss_train={}
            for start in range(0,number_image_train,batch_size):
                end=start+batch_size
                summary_train,loss_train,_=sess.run([summary_op,loss,optimizer],feed_dict={
                    X:(x_train[start:end]/255.),
                    Y1:y_train[0][start:end],
                    Y2:y_train[1][start:end],
                    Y3:y_train[2][start:end]
                })

                train_summary_writer.add_summary(summary_train,epoch)
                train_summary_writer.flush()
                mean_loss_train.append(loss_train)
                print("(start: %s end: %s \t epoch: %s)\t loss: %s "%(start,end,epoch+1,loss_train))

            mean_loss_train=np.mean(mean_loss_train)
            duration=time.time()-start_time
            examples_per_sec=number_image_train/duration
            sec_pec_batch=float(duration)

            mean_loss_valid=[]
            for start in range(0,number_image_valid,batch_size):
                end=start+batch_size
                summary_valid,loss_valid=sess.run([summary_op,loss],feed_dict={
                    X:(x_valid[start:end]/255.),
                    Y1:y_valid[0][start:end],
                    Y2:y_valid[1][start:end],
                    Y3:y_valid[2][start:end]
                })
                validation_summary_writer.add_summary(summary_valid,epoch)
                validation_summary_writer.flush()
                mean_loss_valid.append(loss_valid)
            mean_loss_valid=np.mean(mean_loss_valid)
            print("epoch %s / %s \ttrain_loss: %s,\tvalid_loss: %s" % (
            epoch + 1, epochs, mean_loss_train, mean_loss_valid))

            if best_loss_valid>mean_loss_valid:
                best_loss_valid=mean_loss_valid
                if VOC == True:
                    create_new_folder = PATH + "/save_model/SAVER_MODEL_VOC1"
                else:
                    create_new_folder = PATH + "/save_model/SAVER_MODEL_boat10"
                try:
                    os.mkdir(create_new_folder)
                try:
                    os.mkdir(create_new_folder)
                except OSError:
                    pass

                checkpoint_path=create_new_folder+"/model.ckpt"
                saver.save(sess,checkpoint_path,global_step=epoch)
                print("Model saved in file: %s"%checkpoint_path)

        print("Tuning completed!")

        train_summary_writer.close()
        validation_summary_writer.close()