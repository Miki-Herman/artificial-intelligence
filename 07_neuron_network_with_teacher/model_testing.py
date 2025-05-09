import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from tensorflow.keras import models

class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# load trained model --> exits error if not trained
model = models.load_model('./image_model.keras')

# prints the model summary --> info about layers etc
model.summary()

image_path = "./images"

# loads images from image folder for testing if the model is accurate on image that is not trained on
image_files = [file for file in os.listdir(image_path) if file.lower().endswith(".jpg")]
# shuffle image list to introduce randomness
random.shuffle(image_files)

count = len(image_files)
success = 0

for image_file in image_files:

    # label can be extracted from file name bcs of conventions
    label = image_file.split('.')[0].split('_')[0]

    # loads testing image and standardizes color and image size --> some aren't exactly 32x32
    image = cv.imread(os.path.join(image_path, image_file))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (32, 32))

    # shows image
    plt.imshow(image, cmap=plt.get_cmap('gray'))

    print("========================================================")
    print(f"Calculating prediction for image {image_file} ...")

    prediction = model.predict(np.array([image]) / 255)
    index = np.argmax(prediction)
    predicted_label = class_names[index]

    if label.lower() == predicted_label.lower():
        success += 1

    print(f"Prediction for image: {predicted_label}")
    print(f"Actual label: {label}")

    print("-------------------------------------------------------")

print("##############################################")
print("--------------- Final score ------------------")
print("##############################################")

print(f"Successfully predicted: {success}/{count}")
print(f"Unsuccessfully predicted: {count - success}/{count}")
print(f"Accuracy: {(success/count)*100.0:.2f}%")