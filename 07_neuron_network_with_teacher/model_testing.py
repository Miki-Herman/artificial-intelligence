import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras import models

class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = models.load_model('./image_model.keras')
model.summary()

image_path = "./images"

image_files = [file for file in os.listdir(image_path) if file.lower().endswith(".jpg")]

for image_file in image_files:
    image = cv.imread(os.path.join(image_path, image_file))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (32, 32))

    plt.imshow(image, cmap=plt.get_cmap('gray'))

    print("-------------------------------------------")
    print(f"Calculating prediction for image {image_file} ...")

    prediction = model.predict(np.array([image]) / 255)
    index = np.argmax(prediction)

    print(f"Prediction for image: {class_names[index]}")
    print("-------------------------------------------")
