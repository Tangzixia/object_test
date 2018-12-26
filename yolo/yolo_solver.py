#coding=utf-8


import tensorflow as tf
import numpy as np
import re
import sys
import time

from .solver import Solver

class YOLOSolver(Solver):
    def __init__(self,dataset,net,common_params,solver_params):
        self.moment=float(solver_params['moment'])
        self.learning_rate=float(solver_params['learning_rate'])
        self.batch_size=int(common_params['batch_size'])
        self.height=int(common_params['image_size'])
        self.width=int(common_params['image_size'])
        self.max_objects=int(common_params['max_objects_per_image'])
        self.pretrain_path=str(solver_params['pretrain_model_path'])
        self.train_dir=str(solver_params['train_dir'])
        self.max_iterations=str(solver_params['max_iterations'])

        self.dataset=dataset
        self.net=net
        self.construct_graph()

    def _train(self):
        opt=tf.train.MomentumOptimizer(self.learning_rate,self.moment)
        grads=opt.compute_gradients(self.total_loss)
        apply_gradient_op=opt.apply_gradients(grads,global_step=self.global_step)
        return apply_gradient_op

    def construct_graph(self):
        self.global_step=tf.Variable(0,trainable=False)
        self.images=tf.placeholder(tf.float32,(self.batch_size,self.height,self.width,3))
        self.labels=tf.placeholder(tf.float32,(self.batch_size,self.height,self.width,5))
        self.objects_num=tf.placeholder(tf.int32,(self.batch_size))

        self.predicts=self.net.inference(self.images)
        self.total_loss,self.nilboy=self.net.loss(self.predicts,self.labels,self.objects_num)

        tf.summary.scalar("loss",self.total_loss)
        self.init_op=tf.global_variables_initializer()
        self.train_op=self._train()

    def solver(self):
        saver1=tf.train.Saver(self.net.pretrained_collection,write_version=1)
        saver2=tf.train.Saver(self.net.trainable_collection,write_version=1)

        summary_op=tf.summary.merge_all()
        sess=tf.Session()

        sess.run(self.init_op)
        saver1.restore(sess,self.pretrain_path)

        summary_writer=tf.summary.FileWriter(self.train_dir,sess.graph)

        for step in range(self.max_iterations):
            start_time=time.time()
            np_images,np_labels,np_objects_num=self.dataset.batch()

            _,loss_value,nilboy=sess.run([self.train_op,self.total_loss,self.nilboy])

            duration=time.time()-start_time

            if step%10==0:
                num_examples_per_step=self.dataset.batch_size
                examples_per_sec=num_examples_per_step/duration
                sec_per_batch=float(duration)

                format_str=("%s: step %d, loss=%.2f (%.1f examples/sec; %.3f (sec/batch)")
                print(format_str % (datetime.now(),step,loss_value,examples_per_sec,sec_per_batch))

                sys.stdout.flush()

            if step%100:
                summary_str=sess.run(summary_op,feed_dict={self.images:np_images,self.labels:np_labels,self.objects_num:np_objects_num})
                summary_writer.add_summary(summary_str,step)
            if step%5000==0:
                saver2.save(sess,self.train_dir+"/model.ckpt",global_step=step)
        sess.close()

