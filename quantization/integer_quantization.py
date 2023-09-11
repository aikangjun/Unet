import pathlib
import numpy as np
import tensorflow as tf
import config.configure as cfg
import os
import time
from PIL import Image

converter = tf.lite.TFLiteConverter.from_saved_model(cfg.pb_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_data_gen():
    files_list = []
    labels_list = []
    with open(file=cfg.total_txt,mode='r',encoding='utf-8') as files:
        for file in files:
            files_list.append(os.path.join(r'..\\VOCdevkit\\VOC2012\\JPEGImages',file[:-1]+'.jpg'))
    random_choice = np.random.choice(range(file.__len__()),size=100)
    choice_files = np.array(files_list)[random_choice]

    for source in choice_files:
        image = Image.open(source)
        image = np.array(image.resize(size=cfg.img_szie),dtype='float32')
        image = image/127.5 - 1
        image = np.expand_dims(image,axis=0)
        yield [image]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_integer_model = converter.convert()
tflite_model_dir = pathlib.Path('.\\tflite_model')
tflite_integer_model_file = tflite_model_dir/'integer_model.tflite'
tflite_integer_model_file.write_bytes(tflite_integer_model)

# 测试integer_tflite模型推理时间
interpreter_integer= tf.lite.Interpreter(model_path=str(tflite_integer_model_file))
interpreter_integer.allocate_tensors()
# 在integer_tflite中，输入类型必须为int8
test_image = np.array(Image.open(r'..\VOCdevkit\VOC2012\JPEGImages\2007_000032.jpg').resize(cfg.img_szie),dtype='int8')
test_image = np.expand_dims(test_image,axis=0)

input_index = interpreter_integer.get_input_details()[0]['index']
output_index = interpreter_integer.get_output_details()[0]['index']
start_time = time.process_time()
interpreter_integer.set_tensor(input_index,test_image)
interpreter_integer.invoke()
prediction = interpreter_integer.get_tensor(output_index)
end_time = time.process_time()
print(f'integer_model推理时间{end_time-start_time}')