import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np

CATEGORIES = ['n02085936-Maltese_dog', 'n02088094-Afghan_hound']

def image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Wrong path:', path)
    else:
        new_arr = cv2.resize(img, (60, 60))
        new_arr = np.array(new_arr)
        new_arr = new_arr.reshape(-1, 60, 60, 1)
        return new_arr

model = keras.models.load_model('maltese-dog-vs-afghan-hound.model')
prediction = model.predict([image('test_maltese_dog.jpg')])
print(CATEGORIES[prediction.argmax()])
prediction = model.predict([image('test_afghan_hound.jpg')])
print(CATEGORIES[prediction.argmax()])