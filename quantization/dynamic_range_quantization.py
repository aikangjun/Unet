import pathlib
import numpy as np
import tensorflow as tf
import config.configure as cfg
import time
from PIL import Image

# dynamic range quantization  动态范围量化
# 激活值以浮点存储，在计算时动态量化为8位，处理后去量化为浮点精度
# 权重在训练后量化，激活在推理时动态量化
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=cfg.pb_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # 指定optimizations 属性可以转变tflite量化的方式
tflite_dynamic_model = converter.convert()
tflite_model_dir = pathlib.Path('.\\tflite_model')
tflite_model_dir.mkdir(exist_ok=True,parents=True)
tflite_dynamic_model_file = tflite_model_dir/"dynamic_model.tflite"
tflite_dynamic_model_file.write_bytes(tflite_dynamic_model)

#  测试dynamic_tflite量化推理的时间
test_image = np.array(Image.open(r'..\VOCdevkit\VOC2012\JPEGImages\2007_000032.jpg').resize(cfg.img_szie),dtype='float32')
test_image = np.expand_dims(test_image,axis=0)
interpreter_dynamic = tf.lite.Interpreter(model_path=str(tflite_dynamic_model_file))
interpreter_dynamic.allocate_tensors()
input_index = interpreter_dynamic.get_input_details()[0]['index']
output_index = interpreter_dynamic.get_output_details()[0]['index']

start_time = time.process_time()
interpreter_dynamic.set_tensor(input_index,test_image)
interpreter_dynamic.invoke()
prediction = interpreter_dynamic.get_tensor(output_index)
end_time = time.process_time()
# intel cpu没有对量化推理计算进行优化，arm cpu进行优化后可以加快推理
print(f'dynamic_tflite推理时间{end_time - start_time}')