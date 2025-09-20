# Code in this file is adapted from pytorch/pytorch
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import PIL.Image

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class IMDB_WIKI(VisionDataset):
    """`The IMDB-WIKI Data Set <https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/>`

    TODO: add description.

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

    _URL_MD5 = {
        "data": ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar", "44b7548f288c14397cb7a7bab35ebe14"),
        "meta": ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_meta.tar", "469433135f1e961c9f4c0304d0b5db1e"),
    }
    # TODO: WIKI dataset is not processed yet

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
        self._base_folder = Path(self.root) / "imdb_wiki"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "imdb_crop"

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
        for url, md5 in self._URL_MD5.values():
            download_and_extract_archive(url, download_root=self._base_folder, md5=md5)
