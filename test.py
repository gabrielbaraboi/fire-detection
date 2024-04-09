import tensorflow as tf
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

model_path = 'fires_classification_model.h5'
model = tf.keras.models.load_model(model_path)
classes = ['fire', 'not fire']


def classify_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]

    class_label = classes[class_index]
    return class_label, confidence


if __name__ == "__main__":

    image_path = sys.argv[1]
    class_label, confidence = classify_image(image_path)
    img = cv2.imread(image_path,-1)

    # Display the image with classification result
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Class: {class_label}, Confidence: {confidence:.2f}')
    plt.axis('off')
    plt.show()