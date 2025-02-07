# Code in this file is adapted from TorchIO-project/torchio
# https://github.com/TorchIO-project/torchio/blob/main/src/torchio/datasets/ixi.py

import copy
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Optional, Tuple

import torchio as tio
from torchio.download import download_and_extract_archive, download_url

from torch.utils.data import Dataset
from torchvision.datasets.utils import verify_str_arg


class IXI_TINY(Dataset):
    """IXI Tiny Dataset (T1-weighted MR Images).

    This is a tiny version of the `IXI dataset <https://brain-development.org/ixi-dataset/>`_,
    featured in the main `notebook`_ of `TorchIO <https://github.com/fepegar/torchio>`_.
    It contains 566 :math:`T_1`-weighted brain MR images along with their corresponding
    brain segmentations, all resized to :math:`83 \times 44 \times 55` voxels.

    It can be used as a medical image MNIST.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.

    .. _notebook: https://github.com/fepegar/torchio/blob/main/tutorials/README.md
    """

    _DATA_URL = "https://www.dropbox.com/s/ogxjwjxdv5mieah/ixi_tiny.zip?dl=1"
    _DATA_MD5 = "bfb60f4074283d78622760230bfa1f98"
    _LABEL_URL = "https://github.com/pm25/regression-datasets/raw/refs/heads/main/data/ixi_tiny/meta.zip"
    _LABEL_MD5 = "9ec9c81a4d34d5766d0e8c148d34e9da"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        load_getitem: bool = True,
    ) -> None:
        super().__init__()
        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = Path(root) / "ixi_tiny"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "image"
        self.transform = transform
        self.target_transform = target_transform
        self.load_getitem = load_getitem

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
        image = tio.ScalarImage(image_file)
        # image = copy.deepcopy(image)  # cheap since images not loaded yet
        if self.load_getitem:
            image.load()

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
        """Download the dataset if it does not exist already."""
        if self._check_exists():
            return
        with NamedTemporaryFile(suffix=".zip", delete=True) as f:
            download_and_extract_archive(self._DATA_URL, download_root=self._base_folder, filename=f.name, md5=self._DATA_MD5)
            images_folder = self._base_folder / "ixi_tiny" / "image"
            if images_folder.is_dir():
                images_folder.rename(self._images_folder)
            extracted_folder = self._base_folder / "ixi_tiny"
            if extracted_folder.exists():
                shutil.rmtree(extracted_folder)
        download_url(self._LABEL_URL, root=self._base_folder, md5=self._LABEL_MD5)
