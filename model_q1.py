import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load and preprocess the dataset
# Assuming you have a 'horse_or_human' directory with 'train' and 'test' folders inside
data_dir = r"C:\Users\Harshini\Downloads\Mid-model Exam\Mid-model Exam\horse-or-human\horse-or-human-20241104T062424Z-001\horse-or-human"

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    data_dir ,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    data_dir ,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

test_data = test_datagen.flow_from_directory(
    data_dir ,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# 2. Visualize samples
sample_images, sample_labels = next(train_data)
fig, axes = plt.subplots(1, 5, figsize=(15, 6))
for i, ax in enumerate(axes):
    ax.imshow(sample_images[i])
    ax.set_title("Horse" if sample_labels[i] == 0 else "Human")
    ax.axis('off')
plt.show()

# 3. Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# 4. Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. Train the Model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# 6. Evaluate Training and Validation Performance
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 7. Test the Model
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# 8. Confusion Matrix on Test Data
test_images, test_labels = next(test_data)
pred_labels = (model.predict(test_images) > 0.5).astype("int32")
cm = confusion_matrix(test_labels, pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Horse', 'Human'], yticklabels=['Horse', 'Human'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(test_labels, pred_labels, target_names=['Horse', 'Human']))

# 9. Experiment with Different Parameters
# For example, let's change the learning rate and batch size

def build_model(learning_rate=0.001, dropout_rate=0.5):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Train with new model configuration
new_model = build_model(learning_rate=0.0001, dropout_rate=0.4)
history = new_model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# 10. Apply Overfitting/Underfitting Techniques
# In the above model, dropout and data augmentation were used. You could also experiment with:
# - Early stopping
# - Increased or decreased model complexity

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(
    train_data,
    epochs=20,
    validation_data=val_data,
    callbacks=[early_stopping]
)
