"""
Fusion Model Trainer

Author : Mithun

Trains MangoFusionModel (image + environmental) and saves it to a SEPARATE
checkpoint file from the original image-only model, so
outputs/checkpoints/mango_leaf_classifier.keras is never overwritten.
"""

import tensorflow as tf

from src.models.fusion_model import MangoFusionModel
from src.data.fusion_dataset import FusionDatasetLoader
from src.config.config import (
    CHECKPOINT_DIR,
    FUSION_MODEL_NAME,
    FUSION_HEAD_EPOCHS,
    FUSION_FINETUNE_EPOCHS,
    FUSION_HEAD_LR,
    FUSION_FINETUNE_LR,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FusionTrainer:

    def __init__(self):

        self.dataset = FusionDatasetLoader()

        self.model_builder = MangoFusionModel()

        self.model = None

    def callbacks(self):

        return [

            tf.keras.callbacks.ModelCheckpoint(

                filepath=str(CHECKPOINT_DIR / FUSION_MODEL_NAME),

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

    def train(self, fine_tune=True):

        logger.info("Loading Fusion Dataset (image + environmental)...")
        train_ds, val_ds, test_ds = self.dataset.load()
        logger.info("Fusion Dataset Loaded Successfully.")

        logger.info("Building Fusion Model (reusing trained image branch)...")
        self.model = self.model_builder.build()
        logger.info("Fusion Model Built Successfully.")

        self.model_builder.summary()

        # --------------------------------------------------
        # Phase 1: train env branch + fusion head, image branch frozen
        # --------------------------------------------------

        logger.info("Compiling Fusion Model (Phase 1 - head only)...")

        self.model_builder.compile_head(learning_rate=FUSION_HEAD_LR)

        logger.info(f"Starting Phase 1 Training ({FUSION_HEAD_EPOCHS} epochs)...")

        history_head = self.model.fit(

            train_ds,

            validation_data=val_ds,

            epochs=FUSION_HEAD_EPOCHS,

            callbacks=self.callbacks(),

            verbose=1,

        )

        history_fine = None

        # --------------------------------------------------
        # Phase 2 (optional): fine-tune top of image branch, low LR
        # --------------------------------------------------

        if fine_tune:

            logger.info("Preparing Fine-Tuning of Image Branch...")
            self.model_builder.fine_tune(learning_rate=FUSION_FINETUNE_LR)

            logger.info(
                f"Starting Phase 2 Fine-Tuning ({FUSION_FINETUNE_EPOCHS} epochs)..."
            )

            history_fine = self.model.fit(

                train_ds,

                validation_data=val_ds,

                epochs=FUSION_FINETUNE_EPOCHS,

                callbacks=self.callbacks(),

                verbose=1,

            )

        logger.info("=" * 60)
        logger.info("Fusion Training Completed")
        logger.info("=" * 60)

        loss, accuracy, precision, recall = self.model.evaluate(
            test_ds,
            verbose=1
        )

        logger.info(f"Test Accuracy  : {accuracy:.4f}")
        logger.info(f"Test Precision : {precision:.4f}")
        logger.info(f"Test Recall    : {recall:.4f}")
        logger.info(f"Saved fusion weights to : {CHECKPOINT_DIR / FUSION_MODEL_NAME}")

        return history_head, history_fine
