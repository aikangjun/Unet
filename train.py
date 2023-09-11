# -*- coding:utf-8 -*-
"""
作者：chenyinhui
日期：2023年03月01日
"""
import os.path
import tensorflow as tf
import numpy as np
from _utils.generator import Generator
from _utils.utils import WarmUpCosineDecayScheduler
import config.configure as cfg
from model import CusModel

if __name__ == '__main__':
    gen = Generator(train_txt=cfg.train_txt, val_txt=cfg.val_txt,
                    img_size=cfg.img_szie, batch_szie=cfg.batch_size,
                    num_class=cfg.num_class)

    model = CusModel(num_class=cfg.num_class, weight_decay=cfg.weight_decay,
                  learning_rate=cfg.learning_rate)

    if not os.path.exists(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)

    ckpt = tf.train.Checkpoint(model=model.model_unet,
                               optimizer=model.optimizer)

    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                              directory=cfg.ckpt_path,
                                              max_to_keep=3)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('latest checkpoint restored')

    train_gen = gen.generate(training=True)
    val_gen = gen.generate(training=False)
    train_losses, train_scores, train_acc = [], [], []
    valid_losses, valid_scores, valid_acc = [], [], []
    for epoch in range(cfg.Epoches):

        for i in range(gen.get_train_len()):
            sources, targets = next(train_gen)
            model.train(sources, targets)
            if i % 100 ==0:
                pre,label =model.predict(sources,targets)
                pre.save(f'.\\image\\pre\\train_pre_{epoch}_{i}.jpg')
                label.save(f'.\\image\\label\\train_label_{epoch}_{i}.jpg')

        for i in range(gen.get_val_len()):
            sources, targets = next(val_gen)
            model.validate(sources, targets)
            if i % 10 ==0:
                pre,label =model.predict(sources,targets)
                pre.save(f'.\\image\\val_pre_{epoch}_{i}.jpg')
                label.save(f'.\\image\\val_label_{epoch}_{i}.jpg')

        print(
            f'Epoch {epoch + 1}, '
            f'train_loss:  {model.train_loss.result()}, '
            f'valid_loss: {model.val_loss.result()}, '
            f'train_acc: {model.train_acc.result() * 100}, '
            f'valid_acc: {model.val_acc.result() * 100}, '
            f'train_score:  {model.train_score.result() * 100}, '
            f'valid_score: {model.val_score.result() * 100}')

        train_acc.append(model.train_acc.result().numpy() * 100)
        train_losses.append(model.train_loss.result().numpy())
        train_scores.append(model.train_score.result().numpy() * 100)
        valid_acc.append(model.val_acc.result().numpy() * 100)
        valid_losses.append(model.val_loss.result().numpy())
        valid_scores.append(model.val_score.result().numpy() * 100)

        ckpt_save_path = ckpt_manager.save()

        model.train_loss.reset_states()
        model.val_loss.reset_states()
        model.train_acc.reset_states()
        model.val_acc.reset_states()
        model.train_score.reset_states()
        model.val_score.reset_states()
