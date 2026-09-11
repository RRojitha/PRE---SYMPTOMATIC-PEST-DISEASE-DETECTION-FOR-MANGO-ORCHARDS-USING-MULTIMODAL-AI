"""
Train Mango Leaf Disease Classification Model

Author : Mithun
"""

import tensorflow as tf

from src.training.trainer import Trainer
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
    logger.info("🍃 MANGO LEAF DISEASE CLASSIFICATION")
    logger.info("=" * 70)

    logger.info("Checking GPU...")
    configure_gpu()

    logger.info("Initializing Trainer...")
    trainer = Trainer()

    logger.info("Starting Training Pipeline...")
    trainer.train()

    logger.info("=" * 70)
    logger.info("🎉 Training Finished Successfully")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()