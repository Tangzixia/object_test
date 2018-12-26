#coding=utf-8

import tensorflow as tf
import numpy as np
def generator(batch_size=30,shuffle=True):
    all_x=np.random.rand(60,3,3,3).tolist()
    all_y=np.random.rand(60,1).tolist()
    data=np.array(list(zip(all_x,all_y)))
    # data=np.array(zip(all_x,all_y))
    num_batches=int(len(data)/batch_size)
    while True:
        if shuffle:
            shuffle_index=np.random.permutation(np.arange(len(data)))
            data=data[shuffle_index]

        for batch_num in range(num_batches):
            start_index=batch_num*batch_size
            end_index=min((batch_num+1)*batch_size,len(data))
            yield data[start_index:end_index]
if __name__=="__main__":
    generator=generator()
    for iter in range(1):
        print("------------{0}------------".format(iter))
        batch_data = next(generator)
        print(np.array(batch_data[0][0]))
        # print(np.array(batch_data[0]),np.array(batch_data[1]))

