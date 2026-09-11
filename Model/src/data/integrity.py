"""
Dataset Integrity Checker

Checks:
- Dataset exists
- Class folders exist
- Empty folders
- Corrupted images
- Non-RGB images
- Duplicate images (SHA256)
- Image count per class

Author : Mithun
"""

from pathlib import Path
from PIL import Image
import hashlib

from src.config.config import DATASET_DIR, CLASS_NAMES
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetIntegrityChecker:
    """
    Performs dataset integrity checks before training.
    """

    def __init__(self, dataset_dir: Path = DATASET_DIR):
        self.dataset_dir = Path(dataset_dir)

    def check_dataset_exists(self):
        """Verify dataset folder exists."""

        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset folder not found:\n{self.dataset_dir}"
            )

        logger.info("Dataset folder found.")

    def check_class_folders(self):
        """Verify all expected class folders exist."""

        missing = []

        for cls in CLASS_NAMES:
            folder = self.dataset_dir / cls

            if not folder.exists():
                missing.append(cls)

        if missing:
            raise FileNotFoundError(
                f"Missing class folders:\n{missing}"
            )

        logger.info("All class folders exist.")

    def count_images(self):
        """Count images inside each class."""

        summary = {}

        total = 0

        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        for cls in CLASS_NAMES:

            folder = self.dataset_dir / cls

            images = [
                img for img in folder.iterdir()
                if img.suffix.lower() in extensions
            ]

            summary[cls] = len(images)

            total += len(images)

        logger.info("Dataset Summary")

        for cls, count in summary.items():
            logger.info(f"{cls:20s}: {count}")

        logger.info(f"Total Images : {total}")

        return summary

    def verify_images(self):
        """
        Detect corrupted images.
        """

        logger.info("Checking images...")

        bad_images = []

        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        for cls in CLASS_NAMES:

            folder = self.dataset_dir / cls

            for image_path in folder.iterdir():

                if image_path.suffix.lower() not in extensions:
                    continue

                try:

                    with Image.open(image_path) as img:

                        img.verify()

                except Exception:

                    bad_images.append(image_path)

        if bad_images:

            logger.warning(
                f"Found {len(bad_images)} corrupted images."
            )

            for img in bad_images:
                logger.warning(img)

        else:

            logger.info("No corrupted images found.")

    def verify_rgb(self):
        """
        Report non-RGB images.
        """

        logger.info("Checking image color modes...")

        non_rgb = []

        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        for cls in CLASS_NAMES:

            folder = self.dataset_dir / cls

            for image_path in folder.iterdir():

                if image_path.suffix.lower() not in extensions:
                    continue

                try:

                    with Image.open(image_path) as img:

                        if img.mode != "RGB":
                            non_rgb.append((image_path, img.mode))

                except Exception:
                    pass

        if non_rgb:

            logger.warning("Non-RGB images found:")

            for img, mode in non_rgb:
                logger.warning(f"{img} --> {mode}")

        else:

            logger.info("All images are RGB.")

    def check_duplicates(self):
        """
        Detect duplicate files using SHA256 hash.
        """

        logger.info("Checking duplicate images...")

        hashes = {}

        duplicates = []

        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        for cls in CLASS_NAMES:

            folder = self.dataset_dir / cls

            for image_path in folder.iterdir():

                if image_path.suffix.lower() not in extensions:
                    continue

                with open(image_path, "rb") as f:

                    digest = hashlib.sha256(f.read()).hexdigest()

                if digest in hashes:

                    duplicates.append((image_path, hashes[digest]))

                else:

                    hashes[digest] = image_path

        if duplicates:

            logger.warning(
                f"Found {len(duplicates)} duplicate images."
            )

            for img1, img2 in duplicates:
                logger.warning(f"{img1} == {img2}")

        else:

            logger.info("No duplicate images detected.")

    def run(self):
        """
        Execute every integrity check.
        """

        logger.info("=" * 60)
        logger.info("Starting Dataset Integrity Check")
        logger.info("=" * 60)

        self.check_dataset_exists()

        self.check_class_folders()

        self.count_images()

        self.verify_images()

        self.verify_rgb()

        self.check_duplicates()

        logger.info("=" * 60)
        logger.info("Dataset Integrity Check Completed")
        logger.info("=" * 60)


if __name__ == "__main__":

    checker = DatasetIntegrityChecker()

    checker.run()