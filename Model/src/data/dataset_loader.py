"""
TensorFlow Dataset Loader

Author : Mithun
"""

import tensorflow as tf

from pathlib import Path

from src.config.config import (
    DATASET_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED
)

AUTOTUNE = tf.data.AUTOTUNE


class DatasetLoader:

    def __init__(self):

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def load(self):

        # --------------------------
        # Train + Validation
        # --------------------------

        train = tf.keras.preprocessing.image_dataset_from_directory(

            DATASET_DIR,

            validation_split=0.30,

            subset="training",

            seed=SEED,

            image_size=IMAGE_SIZE,

            batch_size=BATCH_SIZE,

            label_mode="categorical",

            shuffle=True

        )

        val_test = tf.keras.preprocessing.image_dataset_from_directory(

            DATASET_DIR,

            validation_split=0.30,

            subset="validation",

            seed=SEED,

            image_size=IMAGE_SIZE,

            batch_size=BATCH_SIZE,

            label_mode="categorical",

            shuffle=True

        )

        # Split validation dataset into validation and test

        size = tf.data.experimental.cardinality(val_test).numpy()

        val_batches = size // 2

        val = val_test.take(val_batches)

        test = val_test.skip(val_batches)

        # Performance optimization

        train = train.cache().shuffle(1000).prefetch(AUTOTUNE)

        val = val.cache().prefetch(AUTOTUNE)

        test = test.cache().prefetch(AUTOTUNE)

        self.train_ds = train

        self.val_ds = val

        self.test_ds = test

        return train, val, test

    def class_names(self):

        dataset = tf.keras.preprocessing.image_dataset_from_directory(

            DATASET_DIR,

            image_size=IMAGE_SIZE,

            batch_size=BATCH_SIZE

        )

        return dataset.class_names