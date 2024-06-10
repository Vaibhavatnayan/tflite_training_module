  
import numpy as np
import tensorflow as tf

def print_tensor_shapes(interpreter):
    # Get input details
    input_details = interpreter.get_input_details()
    print("Input Tensor Shape:")
    for detail in input_details:
        print(detail['shape'])

    # Get output details
    output_details = interpreter.get_output_details()
    print("\nOutput Tensor Shape:")
    for detail in output_details:
        print(detail['shape'])

def load_tflite_model(model_path):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

if __name__ == "__main__":
    # Path to the TensorFlow Lite model
    model_path = "yolov4-416-fp32.tflite"

    # Load the TFLite model
    interpreter = load_tflite_model(model_path)

    # Print input and output tensor shapes
    print_tensor_shapes(interpreter)
