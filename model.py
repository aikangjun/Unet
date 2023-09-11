from net.Unet import Unet
import tensorflow as tf
import tensorflow.keras as keras
from _utils.utils import calculate_score
from PIL import Image
import numpy as np
import config.configure as cfg

class CusModel(object):
    def __init__(self,
                 num_class: int,
                 weight_decay: float,
                 learning_rate: float,
                 **kwargs):
        super(CusModel, self).__init__(**kwargs)
        self.num_class = num_class
        self.model_unet = Unet(num_class=num_class)

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        self.loss_fn = keras.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.AUTO)
        self.optimizer = keras.optimizers.SGD(learning_rate=learning_rate,decay = 1e-6,momentum=0.9,nesterov=True)

        self.train_loss = keras.metrics.Mean()
        self.val_loss = keras.metrics.Mean()
        self.train_score = keras.metrics.Mean()
        self.val_score = keras.metrics.Mean()
        self.train_acc = keras.metrics.CategoricalAccuracy()
        self.val_acc = keras.metrics.CategoricalAccuracy()

    def train(self, sources, targets):
        # 在一个batch内计算，更新参数
        targets = tf.one_hot(targets, depth=self.num_class)
        with tf.GradientTape() as tape:
            # (b,h,w,21)
            logits = self.model_unet(sources)
            loss = self.loss_fn(targets, logits)
            # 在loss_fn加入l2正则化，进行weight_dency
            # for variable in self.model.trainable_variables:
            #     loss += self.weight_decay * tf.reduce_sum(tf.math.square(variable))

        gradients = tape.gradient(loss, self.model_unet.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.model_unet.trainable_variables))

        self.train_loss(loss)
        self.train_acc(targets, logits)
        self.train_score(calculate_score(targets, logits))

    def validate(self, sources, targets):

        logits = self.model_unet(sources)
        targets = tf.one_hot(targets, depth=self.num_class)
        loss = self.loss_fn(targets, logits)

        self.val_loss(loss)
        self.val_acc(targets, logits)
        self.val_score(calculate_score(targets, logits))

    def predict(self, source, target):
        # source为单张图片的数据
        # (1,512,512,21)
        pre = self.model_unet(source)[2]
        # (512,512)
        pre = tf.argmax(pre,axis=-1)
        label = target[2]

        colors = np.array(cfg.colors)
        # np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
        pre = colors[pre]
        pre = Image.fromarray(np.uint8(pre))

        label = colors[label]
        label = Image.fromarray(np.uint8(label))
        return pre,label

if __name__ == '__main__':
    model = CusModel(21,3,0.01)
    # source = tf.random.normal(shape=(1,256,256,3))
    # target = tf.random.normal(shape=(1,256,256,21))
    source = np.random.randint(low=0,high=256,size=(1,256,256,3))
    target = np.random.randint(low=0,high=256,size=(1,256,256,21))

    pre = np.random.randint(low=0,high=256,size=(256,256))
    pre = np.array(cfg.colors,np.uint8)[np.reshape(pre,[-1])]
