#coding=utf-8

import xml.etree.ElementTree as ET
from os import getcwd

sets=[("2012","train"),("2012","val")]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def convert_annotaion(year,image_id,list_file):
    in_file=open("/home/hp/Code/VOCdevkit/VOC%s/Annotaions/%s.xml"%(year,image_id))
    tree=ET.parse(in_file)
    root=tree.getroot()

    for obj in root.iter("object"):
        difficult=obj.find("difficult").text
        cls=obj.find("name").text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id=classes.index(cls)
        xml_box=obj.find("bndbox")
        b=(int(xml_box.find("xmin").text),int(xml_box.find("ymin").text),int(xml_box.find("xmax").text),int(xml_box.find("ymax").text))
        list_file.write(" "+",".join([str(a) for a in b]+","+str(cls_id)))

wd=getcwd()
for year,image_set in sets:
    image_ids=open("/home/hp/Code/VOCdevkit/VOC%s/ImageSets/%s.txt"%(year,image_set)).strip().split()
    list_file=open("./result/%s_%s.txt"%(year,image_set),'w')

    for image_id in image_ids:
        list_file.write("/home/hp/Code/VOCdevkit/VOC2012/JPEGImages/%s.jpg"%(img_id))
        convert_annotaion(year,image_id,list_file)
        list_file.write("\n")
    list_file.close()

