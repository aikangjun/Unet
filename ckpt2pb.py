
from tensorflow import keras

from net.Unet import Unet
import tensorflow as tf
import config.configure as cfg


model = Unet(num_class=22)
ckpt = tf.train.Checkpoint(model=model)
ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                          directory=cfg.ckpt_path,
                                          max_to_keep=3)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print("加载参数成功")


model.compute_output_shape(input_shape=(1,128,128,3))
model.summary()
model.save(cfg.pb_model_path)