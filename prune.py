import tensorflow_model_optimization as tfmot
import tensorflow as tf
import numpy as np
from net.Unet import Unet
import config.configure as cfg

model = Unet(num_class=cfg.num_class)

ckpt = tf.train.Checkpoint(model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt,
                                          directory=cfg.ckpt_path,
                                          max_to_keep=3)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print('checkpiont加载成功')
model.compute_output_shape(input_shape=(1,128,128,3))
# 进行剪枝
# prune_model_magnitude不支持对自定义模型进行直接剪枝，
# prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
# model_for_pruning = prune_low_magnitude(model)
# model_for_pruning.summary()