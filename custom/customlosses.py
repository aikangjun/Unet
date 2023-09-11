# -*- coding:utf-8 -*-
"""
作者：chenyinhui
日期：2023年02月27日
"""
from custom import *

class ReduceCrossEntropy(losses.Loss):
    # 降维交叉熵损失
    # 使用了l2正则项，进行weight_dency
    # 计算一个batch内的loss
    def __init__(self,
                 num_class:int,
                 **kwargs):
        super(ReduceCrossEntropy, self).__init__(**kwargs)
        self.num_class = num_class
        assert self.reduction in [losses.Reduction.AUTO,
                                  losses.Reduction.SUM]

    def call(self, y_true, y_pred):
        # 定义损失函数 reduction形参,指定降维，tf.losses.Reduction.NONE 不进行降维
        loss_fn = losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
        y_true = tf.one_hot(y_true,depth=self.num_class)
        bacth_size = tf.shape(y_true)[0]

        # 返回的是的一个长度为bacth_size的list，里面存放每个样本的loss
        crossentropy_loss = loss_fn(y_true,y_pred)
        # 如果批量大小为一，直接返回crossentropy的值
        if tf.equal(bacth_size,1):
            return  crossentropy_loss


        # tf.cast(a,dtpye=tf.float32),铸造，将a转为tf.float32数据类型
        # 返回一个索引阈值
        indices_threshold = tf.cast(bacth_size//self.num_class,
                                    dtype=tf.float32)

        # 返回一个bool 的list,如果一个batch中含有的同类数量大于索引阈值，返回True
        # 长度为num_class
        bool_mask =  [tf.reduce_sum(y_true[:,i]) > indices_threshold
                      for i in range(self.num_class)]
        # 取出batch中第一个 同类个数大于indices_threshold 的索引
        indice = tf.argmax(tf.cast(bool_mask,dtype=tf.int32))

        # tf.reduce_any(a) 对a中的元素进行逻辑或or,返回bool
        # 返回一个长度为batch_size 的tf.tensor,如果bool_mask中存在True
        neg_mask = y_true[:,indice] if tf.reduce_any(bool_mask) else tf.zeros(shape=(bacth_size,))
        neg_mask = tf.cast(neg_mask,dtype=tf.bool)


if __name__ == '__main__':
    a = tf.constant([[0,1,0,0,0],[0,0,1,0,0]])
    bool_mask = [tf.reduce_sum(a[:, i]) > 0
                 for i in range(5)]
    indice = tf.argmax(tf.cast(bool_mask, dtype=tf.int32))
    print(indice)
    b = tf.constant([[0.7,0.1,0.2,0,0],[0.1,0.8,0.1,0,0]])
    loss_fn = losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
    loss = loss_fn(a,b)
    print(loss)
    print(tf.reduce_sum(loss))
    indices_threshold= tf.cast(32//21,dtype=tf.float32)
    print(indices_threshold)