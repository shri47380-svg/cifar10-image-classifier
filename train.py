import tensorflow as tf
from tensorflow.keras import layers, models

print("Loading CIFAR-10 dataset...")

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Dataset loaded successfully!")

# Data augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# Create CNN model
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    # Data augmentation
    data_augmentation,

    # CNN layers
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),

    # Convert to 1D
    layers.Flatten(),

    # Fully connected layers
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Starting training...")

# Train model
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1
)

print("Training completed!")

# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Save model
model.save("cifar10_model.keras")

print("==============================")
print("Model saved as cifar10_model.keras")
print("==============================")