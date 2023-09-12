import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Normalize pixel values between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Setup the layers
# Layer 1 - Convolutional 2d layer with relu activation. Input layer for the image
# Layer2 - Convolutional 2d layer
# Layer 3 - Convolutional 2d layer
# Layer 4 - Fully connected dense layer
# Layer 5 - Fully connected dense layer with 10 output nodes to map to the output label prediction
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10)
])


# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# Fit the model to the data using 6 epochs 
history = model.fit(train_images, train_labels, epochs=6,
                    validation_data=(test_images, test_labels))
# Evaluate the model on the test data separated from the dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)
# Print the accuracy
print(f'Test accuracy: {test_acc}')




# Show some predictions using matplotlib
num_images_to_display = 5
test_images_display, test_labels_display = test_images[:num_images_to_display], test_labels[:num_images_to_display]
# Get model predictions
predictions = model.predict(test_images_display)
# labels for cifar10
class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
plt.figure(figsize=(12, 8))
for i in range(num_images_to_display):
    plt.subplot(1, num_images_to_display, i + 1)
    plt.imshow(test_images_display[i])
    # Get prediction
    predicted_label = class_labels[np.argmax(predictions[i])]
    # Get label
    true_label = class_labels[test_labels_display[i][0]]
    # Display the predicted label and true label
    plt.title(f"Predicted: {predicted_label}\nTrue: {true_label}", fontsize=10)
    # Hide the plot axis
    plt.axis("off")

plt.tight_layout()
plt.show()