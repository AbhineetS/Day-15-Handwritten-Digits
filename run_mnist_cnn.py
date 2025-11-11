import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

print("📦 Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Training data shape: {x_train.shape}, Labels: {y_train.shape}")

# Define CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train
print("🚀 Training model (3 epochs)...")
history = model.fit(x_train, y_train, epochs=3, validation_split=0.1, verbose=1)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"✅ Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

# Save model
model.save("mnist_cnn.h5")

# Plot accuracy
plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Training Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_accuracy.png")
plt.close()

# Sample predictions
y_pred = np.argmax(model.predict(x_test[:25]), axis=1)

plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap="gray")
    plt.title(f"Pred: {y_pred[i]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.close()

print("📊 Saved training_accuracy.png and sample_predictions.png")