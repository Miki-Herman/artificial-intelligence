import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from tensorflow.keras import models

# Define class names
class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Define the models to test
model_paths = [
    './distilled_cifar10_model.keras',  # Student model with knowledge distillation
    './baseline_cifar10_model.keras',   # Baseline model without distillation
    './teacher_cifar10_model.keras'     # Teacher model
]

# Path to your test images
image_path = "./images"

# Load image files from the directory
image_files = [file for file in os.listdir(image_path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
# Shuffle image list for randomness
random.shuffle(image_files)

# Function to test a model
def test_model(model_path):
    try:
        # Load the model
        print(f"\nLoading model from {model_path}...")
        model = models.load_model(model_path)

        # If it's the distilled student model that was saved with custom loss function
        # we need to recompile it with standard categorical crossentropy for predictions
        if 'distilled' in model_path:
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

        # Print model summary
        model.summary()

        count = len(image_files)
        success = 0

        # Create a figure to display results
        plt.figure(figsize=(15, 10))

        # Process each image
        for i, image_file in enumerate(image_files):
            # Extract label from filename
            label = image_file.split('.')[0].split('_')[0]

            # Load and preprocess the image
            image = cv.imread(os.path.join(image_path, image_file))
            if image is None:
                print(f"Error loading image {image_file}, skipping...")
                continue

            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = cv.resize(image, (32, 32))

            # Display progress
            print("========================================================")
            print(f"Calculating prediction for image {image_file} ...")

            # Normalize and make prediction
            prediction = model.predict(np.array([image]) / 255, verbose=0)
            index = np.argmax(prediction)
            predicted_label = class_names[index]

            # Check if prediction matches label
            is_correct = label.lower() == predicted_label.lower()
            if is_correct:
                success += 1
                color = 'green'
            else:
                color = 'red'

            # Display results
            print(f"Prediction: {predicted_label} (Confidence: {prediction[0][index]:.4f})")
            print(f"Actual label: {label}")
            print(f"Correct: {'Yes' if is_correct else 'No'}")

            # Plot the image with prediction (max 16 images)
            if i < 16:
                plt.subplot(4, 4, i+1)
                plt.imshow(image)
                plt.title(f"True: {label}", fontsize=10)
                plt.xlabel(f"Pred: {predicted_label}", color=color, fontsize=10)
                plt.xticks([])
                plt.yticks([])

        # Show the figure
        plt.tight_layout()
        plt.savefig(f"{os.path.basename(model_path)}_results.png")
        plt.show()

        # Print final score
        print("\n##############################################")
        print(f"--- RESULTS FOR {os.path.basename(model_path)} ---")
        print("##############################################")
        print(f"Successfully predicted: {success}/{count}")
        print(f"Unsuccessfully predicted: {count - success}/{count}")
        print(f"Accuracy: {(success/count)*100.0:.2f}%")

        return (success/count)*100.0

    except Exception as e:
        print(f"Error testing model {model_path}: {str(e)}")
        return 0.0

# Test each model and collect results
results = {}
for model_path in model_paths:
    try:
        accuracy = test_model(model_path)
        results[os.path.basename(model_path)] = accuracy
    except FileNotFoundError:
        print(f"Model file {model_path} not found, skipping...")

# Compare results
if results:
    print("\n##############################################")
    print("------------- COMPARISON RESULTS -------------")
    print("##############################################")
    for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name}: {accuracy:.2f}%")
else:
    print("\nNo models were successfully tested. Please check the model paths.")