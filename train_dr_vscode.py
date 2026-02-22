import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib.image import imread

# ==============================
# DATASET PATH (LOCAL WINDOWS)
# ==============================
DATASET_PATH = r"H:\My Drive\Colab Notebooks\datasets\gaussian_filtered_images\gaussian_filtered_images"

# ==============================
# IMAGE DATA GENERATOR
# ==============================
datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

valid_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ==============================
# VISUALIZE SAMPLE IMAGES
# ==============================
class_names = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]
n_images_per_class = 2

plt.figure(figsize=(8, 10))

for row_idx, cls in enumerate(class_names):
    cls_dir = os.path.join(DATASET_PATH, cls)
    files = os.listdir(cls_dir)
    sample_files = random.sample(files, min(n_images_per_class, len(files)))

    for col_idx, fname in enumerate(sample_files):
        img_path = os.path.join(cls_dir, fname)
        img = imread(img_path)

        ax = plt.subplot(len(class_names), n_images_per_class,
                         row_idx * n_images_per_class + col_idx + 1)
        ax.imshow(img)
        ax.set_title(cls)
        ax.axis("off")

plt.tight_layout()
plt.show()

# ==============================
# LEARNING RATE SCHEDULER
# ==============================
def lr_rate(epoch, lr):
    if epoch < 10:
        return 0.0001
    elif epoch <= 15:
        return 0.0005
    elif epoch <= 30:
        return 0.0001
    else:
        return lr * (epoch / (1 + epoch))

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_rate)

# ==============================
# MODEL BUILDING
# ==============================
IMG_SIZE = (224, 224)
NUM_CLASSES = 5
UNFREEZE_LAYERS = 20

inputs = Input(shape=(224, 224, 3))
x = layers.Rescaling(1./255)(inputs)

data_augment = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.06),
])

x = data_augment(x)

base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=x)
base_model.trainable = False

gap = layers.GlobalAveragePooling2D()(base_model.output)
gmp = layers.GlobalMaxPooling2D()(base_model.output)
x = layers.Concatenate()([gap, gmp])

x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)

if UNFREEZE_LAYERS > 0:
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# TRAIN
# ==============================
history = model.fit(
    train_data,
    validation_data=valid_data,
    callbacks=[lr_callback],
    epochs=40
)

# ==============================
# SAVE MODEL (LOCAL)
# ==============================
model.save("TRYDR15.keras")
model.save("TRYDR15.h5")

print("Model Saved Successfully!")

# ==============================
# PLOT TRAINING CURVES
# ==============================
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss")
plt.show()

# ==============================
# PREDICT SINGLE IMAGE (LOCAL)
# ==============================
TEST_IMAGE_PATH = r"H:\your_test_image.jpg"   # <-- change this

model = load_model("TRYDR15.h5", compile=False)

img = cv2.imread(TEST_IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, IMG_SIZE)
img = cv2.GaussianBlur(img, (5, 5), 0)

img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
pred_idx = np.argmax(pred)
pred_class = class_names[pred_idx]
confidence = float(pred[0][pred_idx])

plt.imshow(img)
plt.axis("off")
plt.title(f"Predicted: {pred_class} ({confidence:.3f})")
plt.show()

print("Prediction:", pred_class)
print("Confidence:", round(confidence, 3))
train_dr_vscode.py