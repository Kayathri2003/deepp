import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load and preprocess the dataset with automatic split
data_dir = r"C:\Users\Harshini\Downloads\Mid-model Exam\Mid-model Exam\horse-or-human\horse-or-human-20241104T062424Z-001\horse-or-human"

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)  # 70% train, 15% validation, 15% test

# Training data (70% of the dataset)
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
    
)

# Validation data (15% of the dataset)
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Test data (another 15% of the dataset)
# Here, we use a second instance with a different split, but pointing to the same directory.
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)
test_data = test_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# 2. Visualize samples
sample_images, sample_labels = next(train_data)
fig, axes = plt.subplots(1, 5, figsize=(15, 6))
for i, ax in enumerate(axes):
    ax.imshow(sample_images[i])
    ax.set_title("Horse" if sample_labels[i] == 0 else "Human")
    ax.axis('off')
plt.show()

# 3. Load Pre-trained Models and Modify for Binary Classification
def build_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_resnet50_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# 4. Compile, Train and Fine-tune the Models with Hyperparameter Tuning
def train_and_evaluate(model, learning_rate=0.001, batch_size=32):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(
        train_data,
        epochs=5,
        validation_data=val_data,
        batch_size=batch_size
    )
    return history

# Instantiate both models
vgg16_model = build_vgg16_model()
resnet50_model = build_resnet50_model()

# Experiment with different learning rates and batch sizes
history_vgg16 = train_and_evaluate(vgg16_model, learning_rate=0.0001, batch_size=32)
history_resnet50 = train_and_evaluate(resnet50_model, learning_rate=0.0001, batch_size=32)

# Fine-tune both models by unfreezing some layers and retraining with a lower learning rate
vgg16_model.layers[0].trainable = True
resnet50_model.layers[0].trainable = True

# Retrain the models with a lower learning rate after unfreezing
history_vgg16_finetune = train_and_evaluate(vgg16_model, learning_rate=0.00001)
history_resnet50_finetune = train_and_evaluate(resnet50_model, learning_rate=0.00001)

# 5. Evaluate on Test Data and Generate Confusion Matrices
def evaluate_model(model, data):
    test_loss, test_accuracy = model.evaluate(data)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    return test_loss, test_accuracy

# Evaluate both fine-tuned models
print("VGG16 Model:")
evaluate_model(vgg16_model, test_data)
print("ResNet50 Model:")
evaluate_model(resnet50_model, test_data)

# Generate confusion matrix for both models
def plot_confusion_matrix(model, data):
    test_images, test_labels = next(data)
    pred_labels = (model.predict(test_images) > 0.5).astype("int32")
    cm = confusion_matrix(test_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Horse', 'Human'], yticklabels=['Horse', 'Human'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    print(classification_report(test_labels, pred_labels, target_names=['Horse', 'Human']))

print("Confusion Matrix for VGG16 Model:")
plot_confusion_matrix(vgg16_model, test_data)

print("Confusion Matrix for ResNet50 Model:")
plot_confusion_matrix(resnet50_model, test_data)

