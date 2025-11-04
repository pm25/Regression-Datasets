import re
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torchio as tio

from torch.utils.data import Dataset
from torchvision.datasets.utils import verify_str_arg, extract_archive


class ADNI(Dataset):
    """
    ADNI Dataset (T1-weighted MR Images) <https://adni.loni.usc.edu>.

    This class provides a curated subset of the Alzheimer's Disease Neuroimaging Initiative (ADNI)
    dataset, consisting of T1-weighted brain MRI scans and corresponding metadata.
    
    The preprocessing pipeline automatically extracts the earliest available MRI scan for each subject
    (ensuring no duplicate subjects) and generates training and validation splits.

    To prepare the dataset, download the following four archives and place them in ``data/adni``:

    - ``ADNI1_Complete 1Yr 1.5T.zip``
    - ``ADNI1_Complete_1Yr_1.5T_metadata.zip``
    - ``ADNI1_Complete 1Yr 3T.zip``
    - ``ADNI1_Complete_1Yr_3T_metadata.zip``

    Once the files are placed in the correct directory, executing this script will automatically
    extract the archives, process the XML metadata, and generate the dataset splits
    (`train.csv`, `eval.csv`) under ``data/adni/meta``.

    By default, only cognitively normal (CN) subjects are used as the validation data.

    Parameters
    ----------
        root : str
            Root directory where the dataset is stored.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download : bool, optional
            If ``True``, attempts to download the dataset automatically (not supported for ADNI).
            Default is ``False``.

    Notes
    -----
    - The dataset must be downloaded manually due to ADNI's data access restrictions.
    - The preprocessing automatically selects the earliest available MRI scan per subject (no duplicate subject across dataset).
    """

    _data_files = [
        "ADNI1_Complete 1Yr 1.5T.zip",
        "ADNI1_Complete_1Yr_1.5T_metadata.zip",
        "ADNI1_Complete 1Yr 3T.zip",
        "ADNI1_Complete_1Yr_3T_metadata.zip",
    ]
    _LABEL_URL = "https://github.com/pm25/regression-datasets/raw/refs/heads/main/data/adni/meta.zip"
    _LABEL_MD5 = "42778a9abaacf2276bd95707e05f2175"

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
        self._split = verify_str_arg(split, "split", ("train", "val"))
        self._base_folder = Path(root) / "adni"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "t1-weighted"
        self.transform = transform
        self.target_transform = target_transform
        self.load_getitem = load_getitem

        if download:
            self._download()

        if not self._check_extracted():
            self._extract()

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
        return all((self._base_folder / f).exists() for f in self._data_files)

    def _check_extracted(self) -> bool:
        return self._images_folder.exists()

    def _download(self) -> None:
        """Download the dataset if it does not exist already."""
        if self._check_exists():
            return
        raise RuntimeError(
            "ADNI is not publicly downloadable. " f"Please download the files ({self._data_files}) manually and place them in {self._base_folder}."
        )

    def _extract(self) -> None:
        """Extract archives."""
        for file_name in self._data_files:
            archive = self._base_folder / file_name
            print(f"Extracting {archive.name} ...")
            extract_archive(archive, self._base_folder, remove_finished=False)
        self._move_nii_files()

    def get_earliest_file_per_subject(self):
        """Select the earliest scan (smallest series ID) per subject."""
        earliest_files = {}
        pattern = re.compile(r"ADNI_(\d+_S_\d+)_.*_S(\d+)_I\d+\.xml")

        for xml_path in self._extracted_folder.glob("*.xml"):
            if not xml_path.is_file():
                continue
            if "MPR__GradWarp__B1_Correction" not in xml_path.name or "Scaled_2_" in xml_path.name:
                continue

            match = pattern.match(xml_path.name)
            if match:
                subject_id, series_id = match.group(1), int(match.group(2))
                prev = earliest_files.get(subject_id)
                if prev is None or series_id < prev[0]:
                    earliest_files[subject_id] = (series_id, xml_path)

        selected_files = [tup[1] for tup in earliest_files.values()]
        print(f"✅ Selected {len(selected_files)} earliest scan XML files.")
        return selected_files

    def _move_nii_files(self) -> None:
        """Process XML metadata and create train/validation splits."""
        selected_files = self.get_earliest_file_per_subject()
        self._images_folder.mkdir(exist_ok=True, parents=True)
        for xml_path in tqdm(selected_files, desc="Processing metadata"):
            nii_path = self.locate_nii_from_xml(xml_path)
            if nii_path is None:
                print(f"[Warning] No NIfTI found for {xml_path.name}")
                continue
            shutil.copy2(nii_path, self._images_folder)
