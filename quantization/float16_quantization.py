import pathlib
import numpy as np
import tensorflow as tf
import config.configure as cfg
import time
from PIL import Image
# float16 quantization float16量化
converter = tf.lite.TFLiteConverter.from_saved_model(cfg.pb_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_float16_model = converter.convert()
tflite_model_dir = pathlib.Path('.\\tflite_model')
tflite_model_dir.mkdir(exist_ok=True,parents=True)
tflite_float16_model_file = tflite_model_dir/'float16_model.tflite'
tflite_float16_model_file.write_bytes(tflite_float16_model)


# 测试float16_tflite模型推理时间
test_image = np.array(Image.open(r'..\VOCdevkit\VOC2012\JPEGImages\2007_000032.jpg').resize(cfg.img_szie),dtype='float32')
test_image = np.expand_dims(test_image,axis=0)
interpreter_float16 = tf.lite.Interpreter(model_path=str(tflite_float16_model_file))
interpreter_float16.allocate_tensors()
input_index = interpreter_float16.get_input_details()[0]['index']
output_index = interpreter_float16.get_output_details()[0]['index']

start_time = time.process_time()
interpreter_float16.set_tensor(input_index,test_image)
interpreter_float16.invoke()
prediction = interpreter_float16.get_tensor(output_index)
end_time = time.process_time()
print(f'float16_model推理时间{end_time-start_time}')