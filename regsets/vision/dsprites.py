# Code in this file is adapted from pytorch/pytorch
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any, Callable, Optional, Tuple

import PIL.Image

from torchvision.datasets.utils import download_url, download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class DSprites(VisionDataset):
    """`dSprites Dataset <https://github.com/google-deepmind/dsprites-dataset>`

    dSprites is a dataset of 2D shapes procedurally generated from independent latent factors.
    Each image is 64x64 and contains a single shape (square, ellipse, or heart).

    This class filters the dataset to include only heart shapes (shape index 3), and uses
    the orientation value as the label. Samples where the orientation is exactly 2π are
    excluded, since 2π is equivalent to 0 in the circular space.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _DATA_URL_MD5 = (
        "https://raw.githubusercontent.com/google-deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
        "7da33b31b13a06f4b04a70402ce90c2e",
    )
    _META_URL_MD5 = ("https://github.com/pm25/regression-datasets/raw/refs/heads/main/data/dsprites/meta.zip", "ddea3957fcda24fbe6c4a4a5dde8290d")

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
        self._base_folder = Path(self.root) / "dsprites"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"
        self._data_file = self._base_folder / Path(self._DATA_URL_MD5[0]).name

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
        image_file, label = self._file_paths[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

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
        download_url(self._DATA_URL_MD5[0], root=self._base_folder, md5=self._DATA_URL_MD5[1])
        download_and_extract_archive(self._META_URL_MD5[0], download_root=self._base_folder, md5=self._META_URL_MD5[1])
        self._extract_heart_images()

    def _extract_heart_images(self) -> None:
        """Extracts only the 'heart' shape images and saves them as PNGs with metadata."""
        dataset = np.load(self._data_file, allow_pickle=True, encoding="latin1")
        images = dataset["imgs"]
        latents_values = dataset["latents_values"]

        self._images_folder.mkdir(parents=True, exist_ok=True)
        assert len(images) == len(latents_values)
        idx = 0

        for image, latents in tqdm(zip(images, latents_values), total=len(images), dynamic_ncols=True):
            shape = latents[1]
            if shape != 3:  # Only include heart-shaped objects
                continue

            filename = f"{idx:06d}.png"
            idx += 1

            image_uint8 = (image * 255).astype("uint8")
            PIL.Image.fromarray(image_uint8, mode="L").save(self._images_folder / filename)
