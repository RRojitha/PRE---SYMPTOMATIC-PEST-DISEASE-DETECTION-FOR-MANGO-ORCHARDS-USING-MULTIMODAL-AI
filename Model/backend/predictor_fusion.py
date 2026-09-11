"""
Backend predictor for the fusion model (image + environmental sensors).

Mirrors predictor.py's interface but loads mango_fusion_classifier.keras
and additionally takes temperature/humidity/wind speed.
"""

import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

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

FUSION_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "outputs" / "checkpoints" / "mango_fusion_classifier.keras"
)

model = tf.keras.models.load_model(str(FUSION_MODEL_PATH))


def predict_fusion(image, temperature_c: float, humidity_percent: float,
                    wind_speed_kmh: float):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    env = np.array(
        [[temperature_c, humidity_percent, wind_speed_kmh]],
        dtype=np.float32,
    )

    predictions = model.predict([img, env], verbose=0)[0]

    index = np.argmax(predictions)

    confidence = float(predictions[index])

    disease = CLASS_NAMES[index]

    status = "Healthy" if disease == "Healthy" else "Diseased"

    return {
        "status": status,
        "disease": disease,
        "confidence": round(confidence * 100, 2)
    }
