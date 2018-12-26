#coding=utf-8

import os

def read_dict(file):
    with open(file,'r') as f:
        lines=f.readlines()
        lines=[line.strip() for line in lines]
    keys_list=[]
    for line in lines:
        for item in line:
            keys_list.append(item)
    return keys_list


def handle_file(file,dst_file,keys):
    print(file)
    with open(file,'r') as f:
        lines=f.readlines()
        lines=[line.strip() for line in lines]
    for line in lines:
        print(line.split(" ")[0])

def handle_dir(dir,dst_dir,keys):
    files=os.listdir(dir)
    if os.path.exists(dst_dir)==False:
        os.mkdir(dst_dir)
    else:
        pass

    for file in files:
        file_src=os.path.join(dir,file)
        file_dst=os.path.join(dst_dir,file)
        handle_file(file_src,file_dst,keys)


if __name__=="__main__":
    file="C:/Users/Administrator/Desktop/王萍师姐组/新型验证码信息生成/changyong_500.txt"
    dir="/media/hp/Galaxy/2000"
    dst_dir="/media/hp/Galaxy/2000_dst/"
    keys=read_dict(file)
    handle_dir(dir,dst_dir,keys)
