"""
EfficientNetB0 Mango Leaf Disease Classifier

Author : Mithun
"""

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)
from tensorflow.keras.applications import EfficientNetB0

from src.data.augmentation import data_augmentation
from src.config.config import (
    IMAGE_SIZE,
    NUM_CLASSES,
    DROPOUT_1,
    DROPOUT_2,
    DENSE_UNITS,
)


class MangoLeafModel:
    """
    EfficientNetB0 based classifier.
    """

    def __init__(self):

        self.model = None
        self.backbone = None

    def build(self):

        inputs = Input(shape=(*IMAGE_SIZE, 3))

        # -----------------------------
        # Data Augmentation
        # -----------------------------
        x = data_augmentation(inputs)

        # -----------------------------
        # EfficientNet Preprocessing
        # -----------------------------
        x = tf.keras.applications.efficientnet.preprocess_input(x)

        # -----------------------------
        # Backbone
        # -----------------------------
        self.backbone = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_tensor=x,
        )

        self.backbone.trainable = False

        # -----------------------------
        # Classification Head
        # -----------------------------
        x = self.backbone.output

        x = GlobalAveragePooling2D()(x)

        x = Dropout(DROPOUT_1)(x)

        x = Dense(
            DENSE_UNITS,
            activation="relu"
        )(x)

        x = Dropout(DROPOUT_2)(x)

        outputs = Dense(
            NUM_CLASSES,
            activation="softmax",
            name="predictions"
        )(x)

        self.model = Model(
            inputs,
            outputs,
            name="MangoLeafDiseaseClassifier"
        )

        return self.model

    # ----------------------------------------------------
    # Phase 1
    # ----------------------------------------------------

    def compile_head(self):

        self.model.compile(

            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-3
            ),

            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=0.1
            ),

            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

    # ----------------------------------------------------
    # Phase 2
    # ----------------------------------------------------

    def fine_tune(self):

        # Unfreeze only the top layers
        self.backbone.trainable = True

        for layer in self.backbone.layers[:-20]:
            layer.trainable = False

        for layer in self.backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        self.model.compile(

            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-5
            ),

            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=0.1
            ),

            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

    def summary(self):

        self.model.summary()