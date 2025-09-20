# Code in this file is adapted from pytorch/pytorch
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from torchvision.datasets.utils import download_and_extract_archive


class IMDB_WIKI:
    """`The IMDB-WIKI Data Set <https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/>`
    Here only uses the cropped IMDB dataset.

    Args:
        root (string): Root directory of the dataset.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _IMDB_DATA_URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar"
    _IMDB_DATA_MD5 = "44b7548f288c14397cb7a7bab35ebe14"
    _IMDB_LABEL_URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_meta.tar"
    _IMDB_LABEL_MD5 = "469433135f1e961c9f4c0304d0b5db1e"
    # TODO: WIKI dataset is not processed yet

    def __init__(
        self,
        root: str,
        download: bool = False,
    ) -> None:
        self._base_folder = Path(root) / "imdb_wiki"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "imdb_crop"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._process_and_save_meta()

    def _check_exists(self) -> bool:
        return self._images_folder.exists() and self._images_folder.is_dir()

    def _download(self) -> None:
        """Download the dataset if it does not exist already."""
        if self._check_exists():
            return
        download_and_extract_archive(self._IMDB_DATA_URL, download_root=self._base_folder, md5=self._IMDB_DATA_MD5)
        download_and_extract_archive(self._IMDB_LABEL_URL, download_root=self._base_folder, md5=self._IMDB_LABEL_MD5)

    @staticmethod
    def datenum_to_datetime(datenum):
        """
        Convert MATLAB datenum array to Python datetime array.
        Missing/invalid dates are converted to np.nan.
        """
        dt_list = []
        for dn in datenum:
            if np.isnan(dn) or dn < 366:
                dt_list.append(np.nan)
                continue
            ordinal = int(np.floor(dn)) - 366
            frac = dn % 1
            dt_list.append(datetime.fromordinal(ordinal) + timedelta(days=float(frac)))
        return np.array(dt_list, dtype=object)

    def _process_and_save_meta(self) -> None:
        data = sio.loadmat(self._base_folder / "imdb" / "imdb.mat")
        imdb = data["imdb"]

        full_paths = imdb["full_path"][0, 0].flatten()
        full_paths = [str(path[0]) for path in full_paths]
        genders = imdb["gender"][0, 0].flatten()
        dobs = imdb["dob"][0, 0].flatten()
        photo_years = imdb["photo_taken"][0, 0].flatten()

        dob_dt = self.datenum_to_datetime(dobs)
        photo_dt = np.array([datetime(int(y), 7, 1) for y in photo_years])
        ages = np.array([relativedelta(photo, dob).years if not pd.isna(dob) else np.nan for photo, dob in zip(photo_dt, dob_dt)])

        assert len(full_paths) == len(ages) == len(genders)

        raw_meta_df = pd.DataFrame({"rel_path": full_paths, "age": ages, "gender": genders})
        meta_df = raw_meta_df[["rel_path", "age"]].rename(columns={"rel_path": "file_name", "age": "label"})

        meta_df = meta_df.sort_values(by=["file_name"])
        meta_df.label = meta_df.label.astype(float)
        meta_df = meta_df.sample(frac=1, random_state=222).reset_index(drop=True)  # random shuffle

        val_frac = 0.1
        test_frac = 0.2

        n_total = len(meta_df)
        n_val = min(10000, int(val_frac * n_total))
        n_test = min(20000, int(test_frac * n_total))

        val_df = meta_df.iloc[:n_val]
        test_df = meta_df.iloc[n_val : n_val + n_test]
        train_df = meta_df.iloc[n_val + n_test :]

        self._meta_folder.mkdir(parents=True, exist_ok=True)
        raw_meta_df.to_csv(self._meta_folder / "raw_meta.csv", index=False)
        train_df.to_csv(self._meta_folder / "train.csv", index=False)
        val_df.to_csv(self._meta_folder / "val.csv", index=False)
        test_df.to_csv(self._meta_folder / "test.csv", index=False)


if __name__ == "__main__":
    IMDB_WIKI("./data", download=True)
