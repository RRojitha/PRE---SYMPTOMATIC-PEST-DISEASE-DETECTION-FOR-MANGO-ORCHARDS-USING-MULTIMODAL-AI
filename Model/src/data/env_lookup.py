"""
Environmental Reading Lookup

Author : Mithun

The environmental CSV (temperature / humidity / wind speed) is NOT paired
1:1 with leaf images today - it only has readings grouped by disease class.
This module is the single place that decides how an image gets matched to
an environmental reading, so the pairing strategy can be swapped later
without touching the dataset pipeline or model code.

STAND-IN STRATEGY (current):
    For a given class, pick one CSV row belonging to that class at random.
    This is a placeholder until real per-leaf sensor readings exist - it
    lets the fusion model train end-to-end on the shape of the data it will
    eventually receive, but the env branch is only learning "what readings
    are typical for this class", not a true per-image relationship.

FUTURE STRATEGY (real per-image pairing):
    Once each row has an "image_filename" column that matches an actual
    image file, set ENV_IMAGE_FILENAME_COL in src/config/config.py to that
    column name. This module then does an exact filename lookup instead of
    random per-class sampling. That is the ONLY change required - no other
    file needs to be touched.
"""

import numpy as np
import pandas as pd

from src.config.config import (
    ENV_CSV_PATH,
    ENV_FEATURE_COLS,
    ENV_CLASS_NAME_MAP,
    ENV_IMAGE_FILENAME_COL,
)


class EnvLookup:
    """
    Loads the environmental CSV once and hands out feature vectors either
    by exact image filename (real pairing) or by random sample within a
    class (stand-in pairing), depending on ENV_IMAGE_FILENAME_COL.
    """

    def __init__(self, csv_path=ENV_CSV_PATH, seed=42):

        self.df = pd.read_csv(csv_path)

        # Normalize the CSV's spaced class names ("Die Back") to the
        # underscore folder/CLASS_NAMES form ("Die_Back").
        self.df["mango_leaf_class"] = self.df["mango_leaf_class"].map(
            lambda c: ENV_CLASS_NAME_MAP.get(c, c)
        )

        self.rng = np.random.default_rng(seed)

        self.by_filename = ENV_IMAGE_FILENAME_COL is not None

        if self.by_filename:
            # Real per-image pairing path (future).
            self.df = self.df.set_index(ENV_IMAGE_FILENAME_COL)
        else:
            # Stand-in per-class sampling path (current).
            self.by_class = {
                class_name: group[ENV_FEATURE_COLS].to_numpy(dtype=np.float32)
                for class_name, group in self.df.groupby("mango_leaf_class")
            }

    def sample(self, class_name: str, image_filename: str = None) -> np.ndarray:
        """
        Returns a float32 array [temperature_c, humidity_percent,
        wind_speed_kmh] for the given class (and, once real pairing is
        wired up, the given image filename).
        """

        if self.by_filename:
            row = self.df.loc[image_filename]
            return row[ENV_FEATURE_COLS].to_numpy(dtype=np.float32)

        rows = self.by_class[class_name]
        idx = self.rng.integers(0, len(rows))
        return rows[idx]
