# Code in this file is adapted from pytorch/pytorch
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

import pandas as pd
from pathlib import Path

from torchvision.datasets.utils import download_and_extract_archive


class UTKFace:
    """`The UTKFace Data Set <https://susanqq.github.io/UTKFace/>`
    Args:
        root (string): Root directory of the dataset.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "https://huggingface.co/datasets/py97/UTKFace-Cropped/resolve/main/UTKFace.tar.gz"
    _MD5 = "ae1a16905fbd795db921ff1d940df9cc"

    def __init__(
        self,
        root: str,
        download: bool = False,
    ) -> None:
        self._base_folder = Path(root) / "utkface"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "UTKFace"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._process_and_save_meta()

    def _check_exists(self) -> bool:
        return self._images_folder.exists() and self._images_folder.is_dir()

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self._base_folder, md5=self._MD5)

    def _process_and_save_meta(self) -> None:
        rel_paths, ages, genders, races, dates = [], [], [], [], []

        for file_path in Path(self._images_folder).glob("*.jpg"):
            attrs = file_path.name.split(".")[0].split("_")
            if len(attrs) != 4:
                continue
            rel_paths.append(file_path.relative_to(self._base_folder))
            ages.append(attrs[0])
            genders.append(attrs[1])
            races.append(attrs[2])
            dates.append(attrs[3])

        raw_meta_df = pd.DataFrame({"rel_path": rel_paths, "age": ages, "gender": genders, "race": races, "date": dates})
        meta_df = raw_meta_df[["rel_path", "age"]].rename(columns={"rel_path": "file_name", "age": "label"})

        meta_df.file_name = meta_df.file_name.apply(lambda x: Path(x).name)
        meta_df = meta_df.sort_values(by=["file_name"])
        meta_df.label = meta_df.label.astype(float)
        train_df = meta_df.sample(frac=0.8, random_state=222)
        test_df = meta_df.drop(train_df.index)

        self._meta_folder.mkdir(parents=True, exist_ok=True)
        raw_meta_df.to_csv(self._meta_folder / "raw_meta.csv", index=False)
        train_df.to_csv(self._meta_folder / "train.csv", index=False)
        test_df.to_csv(self._meta_folder / "test.csv", index=False)


if __name__ == "__main__":
    UTKFace("./data", download=True)
