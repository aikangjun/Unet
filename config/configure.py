# -*- coding:utf-8 -*-
"""
作者：chenyinhui
日期：2023年03月01日
"""
# generator
train_txt = r'D:\dataset\image\VOCdevkit\VOC2012\ImageSets\Segmentation\train.txt'
val_txt = r'D:\dataset\image\VOCdevkit\VOC2012\ImageSets\Segmentation\val.txt'
total_txt = r'D:\dataset\image\VOCdevkit\VOC2012\ImageSets\Segmentation\trainval.txt'
img_szie = (128, 128)
batch_size = 8

# model
num_class = 22
learning_rate = 1e-2
weight_decay = 5e-3

# train
Epoches =  10
ckpt_path = '.\\checkpoint'
pb_model_path = '..\\pb_model'
quant_model_path = '..\\quant_model'
# predict
colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
          (0, 128, 128),
          (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
          (192, 0, 128),
          (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
          (0, 64, 128),
          (255, 255, 255)]
color_class = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor', 'outline']

# quantization
