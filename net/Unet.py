# -*- coding:utf-8 -*-
"""
作者：chenyinhui
日期：2023年02月28日
"""
from net import *
from custom.customlayers import EncoderBlock,DecoderBlock

class Unet(models.Model):
    def __init__(self,
                 num_class:int,
                 **kwargs):
        super(Unet, self).__init__(**kwargs)

        self.encoderblock = EncoderBlock()
        self.decoderblock = DecoderBlock()
        self.conv_1 = layers.Conv2D(filters=num_class,kernel_size=(3,3),strides=(1,1),padding='same')
        self.softmax = layers.Softmax(axis=-1)

    def call(self, inputs, training=None, mask=None):
        x = self.encoderblock(inputs)
        x = self.decoderblock(x)
        x = self.conv_1(x)
        x = self.softmax(x)
        return x

if __name__ == '__main__':
    import tensorflow as tf
    sources = tf.random.normal(shape=(4,512,512,3))
    model = Unet(num_class=5)
    logits = model(sources)
    logits = tf.reshape(logits,shape=(logits.shape[0],-1,5))
    print(logits.shape)