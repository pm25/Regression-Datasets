# Code in this file is adapted from TorchIO-project/torchio
# https://github.com/TorchIO-project/torchio/blob/main/src/torchio/datasets/ixi.py

import copy
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Optional, Tuple

import torchio as tio
from torchio.download import download_and_extract_archive

from torch.utils.data import Dataset
from torchvision.datasets.utils import verify_str_arg


class IXI(Dataset):
    """IXI Dataset (T1-weighted MR Images).

    The `Information eXtraction from Images (IXI) <https://brain-development.org/ixi-dataset/>`_
    dataset contains "nearly 600 MR images from normal, healthy subjects",
    including "T1, T2 and PD-weighted images, MRA images and Diffusion-weighted
    images (15 directions)". This implementation uses only the T1-weighted images.

    .. note ::
        This data is made available under the
        Creative Commons CC BY-SA 3.0 license.
        If you use it please acknowledge the source of the IXI data, e.g.
        `the IXI website <https://brain-development.org/ixi-dataset/>`_.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.

    .. warning::
        The size of this dataset is multiple GB.
        If you set :attr:`download` to ``True``, it will take some time
        to be downloaded if it is not already present.
    """

    _DATA_URL = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"
    _DATA_MD5 = "34901a0593b41dd19c1a1f746eac2d58"
    _LABEL_URL = "https://github.com/pm25/regression-datasets/raw/refs/heads/main/data/ixi/meta.zip"
    _LABEL_MD5 = "c235e80dd585a7a2702b7b1b24eceede"

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
        self._base_folder = Path(root) / "ixi"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "ixi_t1"
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
        with NamedTemporaryFile(suffix=".tar", delete=True) as f:
            download_and_extract_archive(
                self._DATA_URL, download_root=self._base_folder, filename=f.name, extract_root=self._images_folder, md5=self._DATA_MD5
            )
        download_and_extract_archive(self._LABEL_URL, download_root=self._base_folder, md5=self._LABEL_MD5)
