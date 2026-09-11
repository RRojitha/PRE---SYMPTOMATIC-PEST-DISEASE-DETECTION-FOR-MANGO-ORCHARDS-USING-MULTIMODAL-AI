"""
Model Trainer

Author : Mithun
"""

import tensorflow as tf

from src.models.model import MangoLeafModel
from src.data.dataset_loader import DatasetLoader
from src.config.config import (
    CHECKPOINT_DIR,
    MODEL_NAME,
    HEAD_EPOCHS,
    FINETUNE_EPOCHS,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:

    def __init__(self):

        self.dataset = DatasetLoader()

        self.model_builder = MangoLeafModel()

        self.model = None

    def callbacks(self):

        return [

            tf.keras.callbacks.ModelCheckpoint(

                filepath=str(CHECKPOINT_DIR / MODEL_NAME),

                monitor="val_accuracy",

                save_best_only=True,

                verbose=1,

            ),

            tf.keras.callbacks.EarlyStopping(

                monitor="val_loss",

                patience=8,

                restore_best_weights=True,

                verbose=1,

            ),

            tf.keras.callbacks.ReduceLROnPlateau(

                monitor="val_loss",

                factor=0.5,

                patience=4,

                verbose=1,

                min_lr=1e-7,

            ),

        ]

    def train(self):

        logger.info("Loading Dataset...")
        train_ds, val_ds, test_ds = self.dataset.load()
        logger.info("Dataset Loaded Successfully.")

        

        logger.info("Building EfficientNetB0 Model...")
        self.model = self.model_builder.build()
        logger.info("Model Built Successfully.")

        

        self.model_builder.summary()

        # --------------------------------------------------
        # Phase 1
        # --------------------------------------------------

        logger.info("Compiling Model (Phase 1)...")

        self.model_builder.compile_head()

        logger.info(f"Starting Phase 1 Training ({HEAD_EPOCHS} epochs)...")


        history_head = self.model.fit(

            train_ds,

            validation_data=val_ds,

            epochs=HEAD_EPOCHS,

            callbacks=self.callbacks(),

            verbose=1,

        )

        # --------------------------------------------------
        # Phase 2
        # --------------------------------------------------

        logger.info("Preparing Fine-Tuning...")
        self.model_builder.fine_tune()

        logger.info(f"Starting Phase 2 Fine-Tuning ({FINETUNE_EPOCHS} epochs)...")

        history_fine = self.model.fit(

            train_ds,

            validation_data=val_ds,

            epochs=FINETUNE_EPOCHS,

            callbacks=self.callbacks(),

            verbose=1,

        )

        logger.info("=" * 60)
        logger.info("Training Completed")
        logger.info("=" * 60)

        loss, accuracy, precision, recall = self.model.evaluate(
            test_ds,
            verbose=1
        )

        logger.info(f"Test Accuracy  : {accuracy:.4f}")
        logger.info(f"Test Precision : {precision:.4f}")
        logger.info(f"Test Recall    : {recall:.4f}")

        return history_head, history_fine