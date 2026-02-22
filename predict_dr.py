import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model("diabetic_retinopathy_model.keras")

class_names = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]
IMG_SIZE = (224, 224)

image_path = "test.jpg"  # Put test image in project folder

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, IMG_SIZE)
img = cv2.GaussianBlur(img, (5, 5), 0)

img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
pred_idx = np.argmax(pred)

print("Predicted:", class_names[pred_idx])
print("Confidence:", float(np.max(pred)))