# Code adapted from pytorch/vision
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg


class STL10(VisionDataset):
    """`STL10 dataset wrapper with blurred images and edge density regression labels.

    This class loads a preprocessed version of STL10, where images are blurred using
    Gaussian blur with varying sigma, and edge density (from Canny edge detector) is used
    as a synthetic regression label.

    Args:
        root (str): Root directory of the dataset.
        split (str, optional): "train" or "test". Defaults to "train".
        transform (Callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform (Callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads and processes the dataset.
    """

    _URL_MD5 = [
        ("http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz", "91f7769df0f17e558f3565bffb0c7dfb"),
        ("https://github.com/pm25/regression-datasets/raw/refs/heads/main/data/stl10/meta.zip", "5c5b39ba88cbde780bfc6a0e5faee3c1"),
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "test"))

        self._base_folder = Path(self.root) / "stl10"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"
        self._data_folder = self._base_folder / "stl10_binary"
        self._data_files = [self._data_folder / "train_X.bin"]

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        metadata = pd.read_csv(self._meta_folder / f"{split}.csv")
        self._file_paths = metadata["file_name"].apply(lambda x: self._images_folder / x).to_numpy(dtype="object")
        self._labels = metadata["label"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self._file_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_path = self._file_paths[idx]
        label = self._labels[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

    def _download(self) -> None:
        if self._check_exists():
            return

        for url, md5 in self._URL_MD5:
            download_and_extract_archive(url, download_root=self._base_folder, md5=md5)

        self._extract_images()

    def _blur_and_return_image(
            self, 
            image: Image.Image, 
            idx: int, 
            sigma_mean: float = 2.0, 
            sigma_std: float = 1.0
    ) -> Image.Image:
        """Apply Gaussian blur with sampled sigma and return the blurred image.

        Args:
            image (PIL.Image): Original image.
            idx (int): Index used for deterministic randomness.
            sigma_mean (float): Mean of Gaussian blur sigma.
            sigma_std (float): Stddev of Gaussian blur sigma.

        Returns:
            PIL.Image: Blurred image.
        """
        rng = np.random.default_rng(seed=idx)
        sigma = np.clip(rng.normal(loc=sigma_mean, scale=sigma_std), 0.01, 4.0)
        sigma = np.round(sigma, 2)
        image_np = np.array(image)
        blurred_np = cv2.GaussianBlur(image_np, ksize=(0, 0), sigmaX=sigma)
        return Image.fromarray(blurred_np)

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

    def _extract_images(self) -> None:
        self._images_folder.mkdir(parents=True, exist_ok=True)
        images = self._load_images()

        for idx, image_np in tqdm(enumerate(images), total=len(images), desc="Saving blurred images", dynamic_ncols=True):
            image = Image.fromarray(np.transpose(image_np, (1, 2, 0)))  # to HWC
            blurred_image = self._blur_and_return_image(image, idx)
            filename = f"{idx:04d}.png"
            blurred_image.save(self._images_folder / filename)
