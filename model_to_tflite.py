


import tensorflow as tf
import os
from config import model_dir

model_name = ''

if __name__ == "__main__":
    saved_model_to_tflite_converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(model_dir, model_name))
    # 量化使用，可不需要
    # saved_model_to_tflite_converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    saved_model_tflite = saved_model_to_tflite_converter.convert()
    with open(model_tflite_path) as tf:
        f.write(saved_model_tflite)