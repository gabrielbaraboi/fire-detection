import tensorflow as tf
import numpy as np
import cv2

model_path = 'model.h5'
model = tf.keras.models.load_model(model_path)
classes = ['fire', 'not fire']


def detect_fire(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img, verbose=0)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]

    class_label = classes[class_index]
    return class_label
