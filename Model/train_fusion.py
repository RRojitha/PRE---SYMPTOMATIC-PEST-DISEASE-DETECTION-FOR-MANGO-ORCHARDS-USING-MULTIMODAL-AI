"""
Train Mango Leaf Disease Fusion Model (Image + Environmental Sensors)

Author : Mithun

Mirrors train.py but drives FusionTrainer instead of Trainer, and saves
results to a separate checkpoint file (mango_fusion_classifier.keras).
Does not touch or overwrite the original image-only model/weights.
"""

import tensorflow as tf

from src.training.fusion_trainer import FusionTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def configure_gpu():
    """
    Configure TensorFlow GPU memory growth.
    """

    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        logger.warning("No GPU detected. Training will run on CPU.")
        return

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logger.info(f"GPU Detected : {gpus[0].name}")

    except RuntimeError as e:
        logger.error(e)


def main():

    logger.info("=" * 70)
    logger.info("MANGO LEAF DISEASE FUSION CLASSIFICATION (Image + Environmental)")
    logger.info("=" * 70)

    logger.info("Checking GPU...")
    configure_gpu()

    logger.info("Initializing Fusion Trainer...")
    trainer = FusionTrainer()

    logger.info("Starting Fusion Training Pipeline...")
    trainer.train(fine_tune=True)

    logger.info("=" * 70)
    logger.info("Fusion Training Finished Successfully")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
