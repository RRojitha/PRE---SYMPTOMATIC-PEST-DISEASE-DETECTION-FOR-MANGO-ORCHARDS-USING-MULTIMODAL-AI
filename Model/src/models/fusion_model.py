"""
Image + Environmental Fusion Model

Author : Mithun

Reuses the trained EfficientNetB0 image classifier (src/models/model.py) as
a frozen/fine-tunable embedding branch - it is NOT retrained from scratch.
Its final softmax layer is stripped so it outputs a 256-d embedding, which
is concatenated with an embedding from a small MLP over
[temperature_c, humidity_percent, wind_speed_kmh], followed by a new
classifier head.

The original MangoLeafModel / mango_leaf_classifier.keras are untouched.
"""

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Concatenate,
)

from src.models.model import MangoLeafModel
from src.config.config import (
    IMAGE_SIZE,
    NUM_CLASSES,
    CHECKPOINT_DIR,
    MODEL_NAME,
    ENV_EMBED_UNITS,
    FUSION_DENSE_UNITS,
    FUSION_DROPOUT,
)


class MangoFusionModel:

    def __init__(self):

        self.model = None
        self.image_embedding_model = None

    def _build_image_embedding_branch(self):
        """
        Rebuilds the exact original architecture (same builder as the
        image-only model) and loads the already-trained weights into it.
        The Dense(256, relu) layer right before the softmax head is then
        exposed as the image embedding output, and the softmax layer is
        dropped.
        """

        base = MangoLeafModel()
        base.build()
        base.model.load_weights(str(CHECKPOINT_DIR / MODEL_NAME))

        # Architecture (src/models/model.py): ... -> Dense(256, relu)
        # -> Dropout -> Dense(NUM_CLASSES, softmax, name="predictions").
        # layers[-3] is that Dense(256, relu) embedding layer.
        embedding_layer = base.model.layers[-3]
        assert embedding_layer.output.shape[-1] == 256, (
            "Expected the pre-softmax Dense(256) layer here - "
            "src/models/model.py architecture may have changed."
        )

        self.image_embedding_model = Model(
            inputs=base.model.input,
            outputs=embedding_layer.output,
            name="ImageEmbeddingBranch",
        )

        return self.image_embedding_model

    def build(self):

        image_embedder = self._build_image_embedding_branch()
        # Frozen by default; unfreeze via fine_tune() for phase 2.
        image_embedder.trainable = False

        image_input = Input(shape=(*IMAGE_SIZE, 3), name="image_input")
        image_embedding = image_embedder(image_input)

        # -----------------------------
        # Environmental Branch (MLP)
        # -----------------------------
        env_input = Input(shape=(3,), name="env_input")

        e = Dense(32, activation="relu")(env_input)
        env_embedding = Dense(ENV_EMBED_UNITS, activation="relu")(e)

        # -----------------------------
        # Fusion
        # -----------------------------
        fused = Concatenate(name="fusion_concat")(
            [image_embedding, env_embedding]
        )

        f = Dense(FUSION_DENSE_UNITS, activation="relu")(fused)
        f = Dropout(FUSION_DROPOUT)(f)

        outputs = Dense(
            NUM_CLASSES,
            activation="softmax",
            name="fusion_predictions",
        )(f)

        self.model = Model(
            inputs=[image_input, env_input],
            outputs=outputs,
            name="MangoFusionModel",
        )

        return self.model

    # ----------------------------------------------------
    # Phase 1: train env branch + fusion head, image branch frozen
    # ----------------------------------------------------

    def compile_head(self, learning_rate):

        self.image_embedding_model.trainable = False

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

    # ----------------------------------------------------
    # Phase 2 (optional): fine-tune top of image branch at a low LR
    # ----------------------------------------------------

    def fine_tune(self, learning_rate):

        self.image_embedding_model.trainable = True

        for layer in self.image_embedding_model.layers[:-20]:
            layer.trainable = False

        for layer in self.image_embedding_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

    def summary(self):

        self.model.summary()
