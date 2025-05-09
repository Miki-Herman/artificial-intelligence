import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, datasets

(training_images, training_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# scaling images down
training_images, test_images = training_images / 255.0, test_images / 255.0

class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    plt.subplot(4, 4, i+1)

    # remove axis
    plt.xticks([])
    plt.yticks([])

    plt.imshow(training_images[i], cmap=plt.get_cmap('gray'))
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

# ---------- Model --------------- #
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten()) # --> Transforms 2D into 1D

# output layer
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model training
model.fit(training_images, training_labels, epochs=10, validation_data=(test_images, test_labels))

loss, accuracy = model.evaluate(test_images, test_labels)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

model.save('image_model.keras')


