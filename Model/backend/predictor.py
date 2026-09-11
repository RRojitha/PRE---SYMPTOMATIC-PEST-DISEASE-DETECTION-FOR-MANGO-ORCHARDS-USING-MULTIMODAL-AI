import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

MODEL_PATH = Path("models/mango_model.keras")

CLASS_NAMES = [
    "Anthracnose",
    "Bacterial_Canker",
    "Cutting_Weevil",
    "Die_Back",
    "Gall_Midge",
    "Healthy",
    "Powdery_Mildew",
    "Sooty_Mould",
]

model = tf.keras.models.load_model(r"C:\Users\mithu\OneDrive\Desktop\Final_year_project\Model\outputs\checkpoints\mango_leaf_classifier.keras")


def predict(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img, verbose=0)[0]

    index = np.argmax(predictions)

    confidence = float(predictions[index])

    disease = CLASS_NAMES[index]

    status = "Healthy" if disease == "Healthy" else "Diseased"

    return {
        "status": status,
        "disease": disease,
        "confidence": round(confidence * 100, 2)
    }
