import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ===============================
# YOUR DATASET PATH
# ===============================
DATASET_PATH = r"H:\My Drive\Colab Notebooks\datasets\gaussian_filtered_images\gaussian_filtered_images"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 5

# ===============================
# DATA GENERATOR
# ===============================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

valid_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("Classes Found:", train_data.class_indices)

# ===============================
# MODEL BUILDING
# ===============================
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# ===============================
# COMPILE
# ===============================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================
# TRAIN
# ===============================
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=EPOCHS
)

# ===============================
# SAVE MODEL
# ===============================
model.save("diabetic_retinopathy_model.keras")
print("Model Saved Successfully!")

# ===============================
# PLOT RESULTS
# ===============================
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()