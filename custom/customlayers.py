# -*- coding:utf-8 -*-
"""
作者：chenyinhui
日期：2023年02月27日
"""
from custom import *


class MyConv2D(layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: tuple = (3, 3),
                 strides: tuple = (1, 1),
                 padding: str = 'same',
                 normalize: bool = True,
                 activation='leaky_relu',
                 **kwargs):
        super(MyConv2D, self).__init__(**kwargs)
        self.conv_1 = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                    strides=strides, padding=padding)
        if normalize:
            self.layer_normal = layers.LayerNormalization()

        self.activation = activations.get(activation)

    def call(self, inputs, *args, **kwargs):
        x = self.conv_1(inputs)
        x = self.layer_normal(x)
        x = self.activation(x)
        return x


class ConvBlock(layers.Layer):
    def __init__(self,
                 filters: int,
                 downsample: bool,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.init_conv = MyConv2D(filters=filters)
        self.final_conv = MyConv2D(filters=filters)
        self.downsample = downsample

        if downsample:
            self.max_pooling = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    def call(self, inputs, *args, **kwargs):
        x = self.init_conv(inputs)
        x = self.final_conv(x)
        features = x
        if self.downsample:
            x = self.max_pooling(x)
            return x, features
        return x


class EncoderBlock(layers.Layer):
    def __init__(self,
                 **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)

        self.first_convblock = ConvBlock(filters=64, downsample=True)
        self.sec_convblock = ConvBlock(filters=128, downsample=True)
        self.third_convblock = ConvBlock(filters=256, downsample=True)
        self.fourth_convblock = ConvBlock(filters=512, downsample=True)
        self.final_convblock = ConvBlock(filters=1024, downsample=False)

    def call(self, inputs, *args, **kwargs):
        x, feats1 = self.first_convblock(inputs)
        x, feats2 = self.sec_convblock(x)
        x, feats3 = self.third_convblock(x)
        x, feats4 = self.fourth_convblock(x)
        x = self.final_convblock(x)
        return x, feats1, feats2, feats3, feats4


class DecoderBlock(layers.Layer):
    def __init__(self,
                 **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.first_conv = ConvBlock(filters=512, downsample=False)
        self.sce_conv = ConvBlock(filters=256, downsample=False)
        self.third_conv = ConvBlock(filters=128, downsample=False)
        self.final_conv = ConvBlock(filters=64, downsample=False)

    def call(self, inputs: tuple, *args, **kwargs):
        x, feats1, feats2, feats3, feats4 = inputs

        x = layers.UpSampling2D()(x)
        x = layers.Concatenate(axis=-1)([x, feats4])
        x = self.first_conv(x)

        x = layers.UpSampling2D()(x)
        x = layers.Concatenate(axis=-1)([x, feats3])
        x = self.sce_conv(x)

        x = layers.UpSampling2D()(x)
        x = layers.Concatenate(axis=-1)([x, feats2])
        x = self.third_conv(x)

        x = layers.UpSampling2D()(x)
        x = layers.Concatenate(axis=-1)([x, feats1])
        x = self.final_conv(x)

        return x


if __name__ == '__main__':
    bolck = ConvBlock(filters=32, downsample=True, upsample=False)
    encoder = EncoderBlock()
    a = tf.random.normal(shape=(4, 512, 512, 3))
    out1 = bolck(a)
    # print(out1[1].shape)
    out2 = encoder(a)
    print(out2)
    b = tf.random.normal(shape=(4, 510, 510, 16))
    print(b.shape[0])

    c = layers.Concatenate(axis=-1)([a[:, 0:b.shape[1], 0:b.shape[2], :], b])
    print(c.shape)
