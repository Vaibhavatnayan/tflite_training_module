import tensorflow as tf
from tensorflow.keras import layers, models

def yolo_conv_block(inputs, filters, training=True):
    x = layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x, training=training)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x

def yolo_output_block(inputs, num_anchors, num_classes):
    x = yolo_conv_block(inputs, 512)
    x = layers.Conv2D(num_anchors * (num_classes + 5), (1, 1), strides=(1, 1), padding='same')(x)
    return x

def YOLOv4_tiny(input_shape=(416, 416, 3), num_classes=4, num_anchors=3):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = yolo_conv_block(x, 64)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = yolo_conv_block(x, 128)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = yolo_conv_block(x, 256)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = yolo_conv_block(x, 512)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    x = yolo_conv_block(x, 1024)
    x = yolo_conv_block(x, 1024)
    x = yolo_conv_block(x, 1024)

    output_large = yolo_output_block(x, num_anchors, num_classes)
    
    model = models.Model(inputs, output_large)
    return model

# Example usage:
model = YOLOv4_tiny(input_shape=(416, 416, 3), num_classes=4, num_anchors=3)
model.summary()
