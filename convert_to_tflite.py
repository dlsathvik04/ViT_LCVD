from onnx_tf.backend import prepare
import onnx
import tensorflow as tf

onnx_path = "vit_model.onnx"
onnx_model = onnx.load(onnx_path)
tf_rep = prepare(onnx_model)
tf_model_path = "vit_model_tf"
tf_rep.export_graph(tf_model_path)

converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("vit_model.tflite", "wb") as f:
    f.write(tflite_model)
print(f"Model converted to vit_model.tflite")