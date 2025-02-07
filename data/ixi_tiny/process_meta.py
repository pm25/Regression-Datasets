# Code in this file is adapted from TorchIO-project/torchio
# https://github.com/TorchIO-project/torchio/blob/main/src/torchio/datasets/ixi.py

import shutil
import pandas as pd
from pathlib import Path
from tempfile import NamedTemporaryFile

from torchio.download import download_and_extract_archive, download_url


class IXI_TINY:
    """IXI Tiny Dataset (T1-weighted MR Images).

    This is a tiny version of the `IXI dataset <https://brain-development.org/ixi-dataset/>`_,
    featured in the main `notebook`_ of `TorchIO <https://github.com/fepegar/torchio>`_.
    It contains 566 :math:`T_1`-weighted brain MR images along with their corresponding
    brain segmentations, all resized to :math:`83 \times 44 \times 55` voxels.

    It can be used as a medical image MNIST.

    Args:
        root (string): Root directory of the dataset.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.

    .. _notebook: https://github.com/fepegar/torchio/blob/main/tutorials/README.md
    """

    _DATA_URL = "https://www.dropbox.com/s/ogxjwjxdv5mieah/ixi_tiny.zip?dl=1"
    _DATA_MD5 = "bfb60f4074283d78622760230bfa1f98"
    _LABEL_URL = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls"
    _LABEL_MD5 = "066002afe250e4375d132266fcfceac5"

    def __init__(
        self,
        root: str,
        download: bool = False,
    ) -> None:
        self._base_folder = Path(root) / "ixi_tiny"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "image"

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
        with NamedTemporaryFile(suffix=".zip", delete=True) as f:
            download_and_extract_archive(self._DATA_URL, download_root=self._base_folder, filename=f.name, md5=self._DATA_MD5)
            images_folder = self._base_folder / "ixi_tiny" / "image"
            if images_folder.is_dir():
                images_folder.rename(self._images_folder)
            extracted_folder = self._base_folder / "ixi_tiny"
            if extracted_folder.exists():
                shutil.rmtree(extracted_folder)
        download_url(self._LABEL_URL, root=self._base_folder, md5=self._LABEL_MD5)

    # adapted from: https://github.com/MMIV-ML/fastMONAI/blob/master/fastMONAI/external_data.py
    def _process_ixi_xls(self) -> pd.DataFrame:
        df = pd.read_excel(self._base_folder / "IXI.xls")

        duplicate_subject_ids = df[df.duplicated(["IXI_ID"], keep=False)].IXI_ID.unique()

        for subject_id in duplicate_subject_ids:
            age = df.loc[df.IXI_ID == subject_id].AGE.nunique()
            if age != 1:
                df = df.loc[df.IXI_ID != subject_id]  # Remove duplicates with two different age values

        df = df.drop_duplicates(subset="IXI_ID", keep="first").reset_index(drop=True)

        df["subject_id"] = ["IXI" + str(subject_id).zfill(3) for subject_id in df.IXI_ID.values]
        df = df.rename(columns={"SEX_ID (1=m, 2=f)": "gender"})
        df["age_at_scan"] = df.AGE.round(2)
        df = df.replace({"gender": {1: "M", 2: "F"}})

        img_list = list(self._images_folder.glob("*.nii.gz"))
        for path in img_list:
            subject_id = path.parts[-1].split("-")[0]
            df.loc[df.subject_id == subject_id, "t1_path"] = str(path.relative_to(self._base_folder))

        df = df[["t1_path", "subject_id", "gender", "age_at_scan"]]
        df = df.dropna()
        return df

    def _process_and_save_meta(self) -> None:
        raw_meta_df = self._process_ixi_xls()
        meta_df = raw_meta_df[["t1_path", "age_at_scan"]].rename(columns={"t1_path": "file_name", "age_at_scan": "label"})

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
    IXI_TINY("./data", download=True)
