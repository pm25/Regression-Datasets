# Code adapted from pytorch/vision
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, List
from pathlib import Path
from tqdm.auto import tqdm

from torchvision.datasets.utils import download_and_extract_archive


class STL10:
    """STL10 Dataset with synthetic regression labels based on edge density after Gaussian blur.

    Downloads the STL10 dataset and processes it into blurred images with edge density labels.

    Args:
        root (str): Root directory of the dataset.
        download (bool): If True, downloads the dataset if not found locally.
    """

    _URL_MD5 = [
        ("http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz", "91f7769df0f17e558f3565bffb0c7dfb"),
    ]

    def __init__(self, root: str, download: bool = False) -> None:
        self._base_folder = Path(root) / "stl10"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"
        self._data_folder = self._base_folder / "stl10_binary"
        self._data_files = [self._data_folder / "train_X.bin"]

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Use download=True to download it.")

        self._process_and_save_meta()

    def _check_exists(self) -> bool:
        return all(file_path.exists() for file_path in self._data_files)

    def _download(self) -> None:
        if self._check_exists():
            return
        for url, md5 in self._URL_MD5:
            download_and_extract_archive(url, download_root=self._base_folder, md5=md5)

    def _blurred_and_calc_edge_density(
        self,
        image: Image.Image,
        idx: int,
        sigma_mean: float = 2.0,
        sigma_std: float = 1.0
    ) -> Tuple[Image.Image, float, float]:
        """Apply Gaussian blur and compute edge density.

        Args:
            image (PIL.Image): Original image.
            idx (int): Index used for deterministic randomness.
            sigma_mean (float): Mean of Gaussian blur sigma.
            sigma_std (float): Stddev of Gaussian blur sigma.

        Returns:
            Tuple: (blurred_image, edge_density, sigma)
        """
        rng = np.random.default_rng(seed=idx)
        sigma = np.clip(rng.normal(loc=sigma_mean, scale=sigma_std), 0.01, 4.0)
        sigma = np.round(sigma, 2)

        # Apply Gaussian blur
        image_np = np.array(image)
        blurred_np = cv2.GaussianBlur(image_np, ksize=(0, 0), sigmaX=sigma)
        blurred_pil = Image.fromarray(blurred_np)

        # Compute edge density from grayscale image
        gray_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_np, threshold1=50, threshold2=150)
        edge_density = edges.mean()

        return blurred_pil, edge_density, sigma

    def _load_images(self) -> np.ndarray:
        """Load raw binary STL10 data and reshape it into images.

        Returns:
            images (np.ndarray): Array of shape (N, 3, 96, 96)
        """
        with open(self._data_files[0], "rb") as f:
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))  # (N, 3, 96, 96) to (N, 3, 96, 96)
        return images

    def _process_and_save_meta(self) -> None:
        images = self._load_images()

        self._images_folder.mkdir(exist_ok=True, parents=True)
        self._meta_folder.mkdir(exist_ok=True, parents=True)

        metadata: List[dict] = []

        for idx, image_np in tqdm(enumerate(images), total=len(images), desc="Processing images", dynamic_ncols=True):
            pil_image = Image.fromarray(np.transpose(image_np, (1, 2, 0)))  # (3, 96, 96) → (96, 96, 3)
            blurred_image, edge_density, sigma = self._blurred_and_calc_edge_density(pil_image, idx)
            filename = f"{idx:04d}.png"
            blurred_image.save(self._images_folder / filename)
            metadata.append({"file_name": filename, "label": edge_density, "sigma": sigma})

        raw_meta_df = pd.DataFrame(metadata).astype({"label": np.float32})
        meta_df = raw_meta_df[["file_name", "label"]]

        train_df = meta_df.sample(n=2000, replace=False, random_state=222)
        meta_df = meta_df.drop(train_df.index)
        val_df = meta_df.sample(n=1000, replace=False, random_state=222)
        test_df = meta_df.drop(val_df.index)

        raw_meta_df.to_csv(self._meta_folder / "raw_meta.csv", index=False)
        train_df.to_csv(self._meta_folder / "train.csv", index=False)
        val_df.to_csv(self._meta_folder / "val.csv", index=False)
        test_df.to_csv(self._meta_folder / "test.csv", index=False)


if __name__ == "__main__":
    STL10("./data", download=True)
