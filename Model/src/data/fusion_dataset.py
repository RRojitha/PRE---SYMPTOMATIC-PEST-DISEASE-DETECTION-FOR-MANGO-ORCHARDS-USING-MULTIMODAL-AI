"""
Fusion Dataset Loader (Image + Environmental Sensors)

Author : Mithun

Builds a tf.data pipeline that yields ((image, env_features), one_hot_label)
triples, pairing every leaf image with an environmental reading via
EnvLookup (see src/data/env_lookup.py for the pairing strategy).

This is separate from src/data/dataset_loader.py (the original image-only
loader), which is left untouched.
"""

import numpy as np
import tensorflow as tf

from pathlib import Path

from src.config.config import (
    DATASET_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
    CLASS_NAMES,
    NUM_CLASSES,
)
from src.data.env_lookup import EnvLookup

AUTOTUNE = tf.data.AUTOTUNE


def _list_image_paths_and_labels():
    """
    Walks DATASET_DIR/<ClassName>/*.jpg and returns parallel lists of
    (file_path, class_index, file_name) sorted deterministically.
    """

    paths, labels, filenames = [], [], []

    for class_idx, class_name in enumerate(CLASS_NAMES):

        class_dir = Path(DATASET_DIR) / class_name

        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            paths.append(str(img_path))
            labels.append(class_idx)
            filenames.append(img_path.name)

    return paths, labels, filenames


class FusionDatasetLoader:

    def __init__(self, env_csv_seed=SEED):

        self.env_lookup = EnvLookup(seed=env_csv_seed)

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def _load_image(self, path):

        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, IMAGE_SIZE)
        image.set_shape((*IMAGE_SIZE, 3))
        return image

    def _sample_env(self, class_idx, filename):
        """
        Wrapped in tf.py_function since env sampling (random choice / CSV
        lookup) happens in numpy/pandas, not in the TF graph.
        """

        class_name = CLASS_NAMES[int(class_idx.numpy())]
        fname = filename.numpy().decode("utf-8")
        return self.env_lookup.sample(class_name, image_filename=fname)

    def _make_example(self, path, label, filename):

        image = self._load_image(path)

        env = tf.py_function(
            func=self._sample_env,
            inp=[label, filename],
            Tout=tf.float32,
        )
        env.set_shape((3,))

        one_hot = tf.one_hot(label, NUM_CLASSES)

        return (image, env), one_hot

    def load(self):

        paths, labels, filenames = _list_image_paths_and_labels()

        paths_t = tf.constant(paths)
        labels_t = tf.constant(labels, dtype=tf.int32)
        filenames_t = tf.constant(filenames)

        full_ds = tf.data.Dataset.from_tensor_slices(
            (paths_t, labels_t, filenames_t)
        )

        full_ds = full_ds.shuffle(
            buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=False
        )

        n = len(paths)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        train_files = full_ds.take(n_train)
        remaining = full_ds.skip(n_train)
        val_files = remaining.take(n_val)
        test_files = remaining.skip(n_val)

        def build(ds, training):
            ds = ds.map(self._make_example, num_parallel_calls=AUTOTUNE)
            if training:
                ds = ds.shuffle(1000, seed=SEED)
            ds = ds.batch(BATCH_SIZE)
            ds = ds.prefetch(AUTOTUNE)
            return ds

        self.train_ds = build(train_files, training=True)
        self.val_ds = build(val_files, training=False)
        self.test_ds = build(test_files, training=False)

        return self.train_ds, self.val_ds, self.test_ds
