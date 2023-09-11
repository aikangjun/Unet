# -*- coding:utf-8 -*-
"""
作者：chenyinhui
日期：2023年02月27日
"""
import numpy as np
from PIL import Image
import os


class Generator:
    def __init__(self,
                 train_txt: str,
                 val_txt: str,
                 img_size: tuple,
                 batch_szie: int,
                 num_class: int,
                 **kwargs
                 ):

        self.train_txt = train_txt
        self.val_txt = val_txt
        self.img_size = img_size
        self.batch_size = batch_szie
        self.num_class = num_class

        self.get_train_file()
        self.get_val_file()

    def get_train_file(self):
        train_source_list = []
        train_target_list = []
        with open(file=self.train_txt, mode='r', encoding='utf-8') as files:
            for file in files:
                source_name = os.path.join(r'D:\dataset\image\VOCdevkit\VOC2012\JPEGImages', file[:-1] + '.jpg')
                train_source_list.append(source_name)
                target_name = os.path.join(r'D:\dataset\image\VOCdevkit\VOC2012\SegmentationClass', file[:-1] + '.png')
                train_target_list.append(target_name)
        self.train_sources_files = train_source_list
        self.train_targets_files = train_target_list

    def get_val_file(self):
        val_source_list = []
        val_target_list = []
        with open(file=self.val_txt, mode='r', encoding='utf-8') as files:
            for file in files:
                source_name = os.path.join(r'D:\dataset\image\VOCdevkit\VOC2012\JPEGImages', file[:-1] + '.jpg')
                val_source_list.append(source_name)
                target_name = os.path.join(r'D:\dataset\image\VOCdevkit\VOC2012\SegmentationClass', file[:-1] + '.png')
                val_target_list.append(target_name)
        self.val_sources_files = val_source_list
        self.val_targets_files = val_target_list

    def get_train_len(self):
        if not self.train_sources_files.__len__() % self.batch_size:
            return self.train_sources_files.__len__() // self.batch_size
        if self.train_sources_files.__len__() % self.batch_size:
            return self.train_sources_files.__len__() // self.batch_size + 1

    def get_val_len(self):
        if not self.val_sources_files.__len__() % self.batch_size:
            return self.val_sources_files.__len__() // self.batch_size
        if self.val_sources_files.__len__() % self.batch_size:
            return self.val_sources_files.__len__() // self.batch_size + 1

    def generate(self, training: bool = True):
        while True:
            if training:
                source_files = self.train_sources_files.copy()
                target_files = self.train_targets_files.copy()
            else:
                source_files = self.val_sources_files.copy()
                target_files = self.val_targets_files.copy()
            source = []
            target = []
            for i, (source_file, target_file) in enumerate(zip(source_files, target_files)):
                img = Image.open(source_file)
                img = np.array(img.resize(self.img_size),dtype="float32")
                img = img / 127.5 - 1

                label = Image.open(target_file)
                label = np.array(label.resize(self.img_size))
                label[label >= self.num_class] = self.num_class - 1

                source.append(img)
                target.append(label)

                if source.__len__() % self.batch_size == 0 or (i + 1) == source_files.__len__():
                    annotation_sources = source.copy()
                    annotation_targets = target.copy()
                    source.clear()
                    target.clear()
                    yield np.array(annotation_sources), np.array(annotation_targets)


