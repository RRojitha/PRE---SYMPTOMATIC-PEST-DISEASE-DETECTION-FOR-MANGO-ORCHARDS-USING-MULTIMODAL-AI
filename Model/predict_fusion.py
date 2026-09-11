"""
Fusion Model Inference (Image + Environmental Sensors)

Author : Mithun

Loads the trained fusion model and predicts a disease class from a leaf
image plus temperature / humidity / wind speed readings.

Usage (CLI):
    python predict_fusion.py <image_path> <temperature_c> <humidity_percent> <wind_speed_kmh>

Example:
    python predict_fusion.py "C:\\path\\to\\leaf.jpg" 26.5 85.0 8.2
"""

import sys

import numpy as np
from PIL import Image

import tensorflow as tf

from src.config.config import (
    CHECKPOINT_DIR,
    FUSION_MODEL_NAME,
    IMAGE_SIZE,
    CLASS_NAMES,
)

_model = None


def _get_model():

    global _model

    if _model is None:
        _model = tf.keras.models.load_model(
            str(CHECKPOINT_DIR / FUSION_MODEL_NAME)
        )

    return _model


def predict(image_path: str, temperature_c: float, humidity_percent: float,
            wind_speed_kmh: float) -> dict:
    """
    Predicts the disease class for a single leaf image + environmental
    reading.

    Returns
    -------
    dict with keys: status, disease, confidence (0-100, rounded to 2 dp)
    """

    model = _get_model()

    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    env_array = np.array(
        [[temperature_c, humidity_percent, wind_speed_kmh]],
        dtype=np.float32,
    )

    predictions = model.predict([img_array, env_array], verbose=0)[0]

    index = int(np.argmax(predictions))
    confidence = float(predictions[index])
    disease = CLASS_NAMES[index]
    status = "Healthy" if disease == "Healthy" else "Diseased"

    return {
        "status": status,
        "disease": disease,
        "confidence": round(confidence * 100, 2),
    }


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print(
            "Usage: python predict_fusion.py <image_path> <temperature_c> "
            "<humidity_percent> <wind_speed_kmh>"
        )
        sys.exit(1)

    image_path = sys.argv[1]
    temperature_c = float(sys.argv[2])
    humidity_percent = float(sys.argv[3])
    wind_speed_kmh = float(sys.argv[4])

    result = predict(image_path, temperature_c, humidity_percent, wind_speed_kmh)

    print(result)
