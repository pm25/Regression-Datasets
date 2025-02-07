# Code in this file is adapted from TorchIO-project/torchio
# https://github.com/TorchIO-project/torchio/blob/main/src/torchio/datasets/ixi.py

import pandas as pd
from pathlib import Path
from tempfile import NamedTemporaryFile

from torchio.download import download_and_extract_archive, download_url


class IXI:
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
    _LABEL_URL = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls"
    _LABEL_MD5 = "066002afe250e4375d132266fcfceac5"

    def __init__(
        self,
        root: str,
        download: bool = False,
    ) -> None:
        self._base_folder = Path(root) / "ixi"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "ixi_t1"

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
        with NamedTemporaryFile(suffix=".tar", delete=True) as f:
            download_and_extract_archive(
                self._DATA_URL, download_root=self._base_folder, filename=f.name, extract_root=self._images_folder, md5=self._DATA_MD5
            )
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
    IXI("./data", download=True)
