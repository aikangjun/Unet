import pathlib
import numpy as np
import tensorflow as tf
import config.configure as cfg
# 转为tflite模型，不进行量化
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=cfg.pb_model_path)
tflite_model = converter.convert()
tflite_model_dir = pathlib.Path('.\\tflite_model')
tflite_model_dir.mkdir(exist_ok=True,parents=True)
tflite_model_file = tflite_model_dir/"tflite_model.tflite"
tflite_model_file.write_bytes(tflite_model) # 将tflite_quant_model(为bytes数据)写到tflite文件中

# 加载模型进入interpreter 测试tflite的推理时间
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
import time
from PIL import Image
test_image = np.array(Image.open(r'C:\Users\chen\Desktop\zvan\Unet-main\VOCdevkit\VOC2012\JPEGImages\2007_000032.jpg').resize(cfg.img_szie),dtype='float32')
test_image = np.expand_dims(test_image,axis=0)
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

start_time = time.process_time()
interpreter.set_tensor(input_index,test_image)
interpreter.invoke()
prediction = interpreter.get_tensor(output_index)
end_time = time.process_time()
print(f'tflite推理时间为{end_time-start_time}')