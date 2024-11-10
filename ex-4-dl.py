#### Lab Exercise 4:
# 1.	Follow the steps for a deep learning model.
    # i.	Load a small dataset of any application of your choice. 
    # ii.	Split the dataset in different ratios (90:10, 80:20,70:30,60:40,50:50) and compare the results.
    # iii.	Implement any 5 models of neural network architecture by initializing the weights and biases.
    # iv.	On trial method use various loss function, optimizers, learning rate, batch sizes and compare.
    # v.	Train the model on the training set.
    # vi.	Monitor the training loss and accuracy after each epoch and graph them.
    # vii.	Evaluate the model on the test set.
    # viii.	Report the accuracy and loss on the test data.
# 2.	Implement techniques like dropout or L2 regularization to prevent overfitting. Compare the performance with and without regularization.
# 3.	Write a report on the training and validation accuracy/loss over epochs and visualize the best test results.

####




import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image

# Load VGG16 model pre-trained on ImageNet
model = VGG16(weights='imagenet', include_top=False)

# Load and preprocess an image
img_path = 'image.jpg'  # Replace with your image path
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Scale to [0, 1]

# Get the output of the first convolutional layer
layer_outputs = model.layers[1].output
feature_map_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

# Get feature maps 
feature_maps = feature_map_model.predict(img_array)

# Visualize the first few feature maps
num_feature_maps = feature_maps.shape[-1]
plt.figure(figsize=(12, 12))
for i in range(min(6, num_feature_maps)):  # Display first 6 feature maps
    plt.subplot(2, 3, i + 1)
    plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
    plt.axis('off')
plt.suptitle('Feature Maps from First Convolutional Layer')
plt.show()

# Get the weights of the first convolutional layer
filters, biases = model.layers[1].get_weights()

# Normalize filter values to 0-1 for better visualization
filters = (filters - filters.min()) / (filters.max() - filters.min())

# Plot the filters
plt.figure(figsize=(12, 12))
for i in range(min(6, filters.shape[3])):  # Display first 6 filters
    plt.subplot(2, 3, i + 1)
    plt.imshow(filters[:, :, :, i], cmap='viridis')
    plt.axis('off')
plt.suptitle('Filters from First Convolutional Layer')
plt.show()
