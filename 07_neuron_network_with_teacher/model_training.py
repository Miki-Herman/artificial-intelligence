import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, applications

# Load and preprocess data
(training_images, training_labels), (test_images, test_labels) = datasets.cifar10.load_data()
training_images = training_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# One-hot encode labels
num_classes = 10
training_labels_one_hot = tf.keras.utils.to_categorical(training_labels, num_classes)
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes)

class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Visualize some examples
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i])
    plt.xlabel(class_names[training_labels[i][0]])
plt.tight_layout()
plt.show()

# Define the distillation loss function
def knowledge_distillation_loss(alpha=0.1, temperature=5.0):
    def loss_function(y_true, y_pred):
        # Extract the true labels and soft targets from y_true
        # First 10 values are true labels, next 10 are teacher's soft targets
        true_labels = y_true[:, :num_classes]
        soft_targets = y_true[:, num_classes:]

        # Standard categorical crossentropy for true labels
        ce_loss = tf.keras.losses.categorical_crossentropy(true_labels, y_pred)

        # KL divergence for soft targets (with temperature scaling)
        kd_loss = tf.keras.losses.kullback_leibler_divergence(
            tf.nn.softmax(soft_targets / temperature),
            tf.nn.softmax(y_pred / temperature)
        ) * (temperature ** 2)

        # Combined loss
        return (1 - alpha) * ce_loss + alpha * kd_loss

    return loss_function

# ---------- TEACHER MODEL (ResNet50) ---------- #
def create_teacher_model():
    # Using a pre-trained model with higher capacity
    # For CIFAR-10 (32x32 images), we'll use a smaller model
    base_model = applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(32, 32, 3)
    )

    # Freeze the base model
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ---------- STUDENT MODEL (Your CNN) ---------- #
def create_student_model():
    model = models.Sequential([
        # Input layer - explicit input shape
        layers.Input(shape=(32, 32, 3)),

        # Your original CNN architecture
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),

        # Enhanced capacity and regularization
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Create and train the teacher model
print("Training the teacher model...")
teacher_model = create_teacher_model()
teacher_model.summary()

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# Model checkpoint callback
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_teacher_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train the teacher model
history_teacher = teacher_model.fit(
    data_augmentation(training_images),
    training_labels_one_hot,
    batch_size=64,
    epochs=20,
    validation_data=(test_images, test_labels_one_hot),
    callbacks=[early_stopping, checkpoint]
)

# Generate teacher's soft predictions
print("Generating teacher's predictions...")
teacher_predictions = teacher_model.predict(training_images)
teacher_test_predictions = teacher_model.predict(test_images)

# Create combined labels with ground truth and teacher predictions
y_train_combined = np.concatenate([training_labels_one_hot, teacher_predictions], axis=1)
y_test_combined = np.concatenate([test_labels_one_hot, teacher_test_predictions], axis=1)

# Create and train the student model
print("Training the student model with knowledge distillation...")
student_model = create_student_model()
student_model.summary()

# Compile student model with knowledge distillation loss
student_model.compile(
    optimizer='adam',
    loss=knowledge_distillation_loss(alpha=0.5, temperature=5.0),
    metrics=['accuracy']
)

# Train student model
history_student = student_model.fit(
    data_augmentation(training_images),
    y_train_combined,
    batch_size=64,
    epochs=50,
    validation_data=(test_images, y_test_combined),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_student_model.keras',
                                           monitor='val_accuracy',
                                           save_best_only=True,
                                           verbose=1)
    ]
)

# Recompile student model for evaluation with standard categorical crossentropy
student_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Evaluate the student model
test_loss, test_accuracy = student_model.evaluate(test_images, test_labels_one_hot)
print(f"Student model test accuracy: {test_accuracy:.4f}")

# Train a baseline model without knowledge distillation for comparison
print("\nTraining a baseline model without distillation for comparison...")
baseline_model = create_student_model()
baseline_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_baseline = baseline_model.fit(
    data_augmentation(training_images),
    training_labels_one_hot,
    batch_size=64,
    epochs=50,
    validation_data=(test_images, test_labels_one_hot),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_baseline_model.keras',
                                           monitor='val_accuracy',
                                           save_best_only=True,
                                           verbose=1)
    ]
)

baseline_loss, baseline_accuracy = baseline_model.evaluate(test_images, test_labels_one_hot)
print(f"Baseline model test accuracy: {baseline_accuracy:.4f}")
print(f"Student model test accuracy: {test_accuracy:.4f}")
print(f"Improvement: {(test_accuracy - baseline_accuracy) * 100:.2f}%")

# Plot training history
plt.figure(figsize=(15, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history_student.history['accuracy'], label='Student Training')
plt.plot(history_student.history['val_accuracy'], label='Student Validation')
plt.plot(history_baseline.history['accuracy'], label='Baseline Training')
plt.plot(history_baseline.history['val_accuracy'], label='Baseline Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history_student.history['loss'], label='Student Training')
plt.plot(history_student.history['val_loss'], label='Student Validation')
plt.plot(history_baseline.history['loss'], label='Baseline Training')
plt.plot(history_baseline.history['val_loss'], label='Baseline Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()

# Save the final models
teacher_model.save('teacher_cifar10_model.keras')
student_model.save('distilled_cifar10_model.keras')
baseline_model.save('baseline_cifar10_model.keras')

# Make predictions with the student model
predictions = student_model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_labels.reshape(-1)

# Display some predictions
plt.figure(figsize=(12, 12))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i])

    predicted_label = class_names[predicted_classes[i]]
    true_label = class_names[true_classes[i]]

    if predicted_classes[i] == true_classes[i]:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel(f"{predicted_label} ({true_label})", color=color)
plt.tight_layout()
plt.show()