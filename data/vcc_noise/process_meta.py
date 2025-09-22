# Code in this file is adapted from pytorch/pytorch
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

import os
import shutil
import pandas as pd
from pathlib import Path

from torchvision.datasets.utils import download_and_extract_archive


class VCC_NOISE:
    """`The VCC Data Set <https://zenodo.org/records/10691660>`
    Args:
        root (string): Root directory of the dataset.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _BVCC_URL = ("https://zenodo.org/records/10691660/files/main.tar.gz", "ba4b896801282ca0eae37b9fd81ed94c")

    def __init__(
        self,
        root: str,
        download: bool = False,
    ) -> None:
        self._base_folder = Path(root) / "vcc_noise"
        self._meta_folder = self._base_folder / "meta"
        self._audio_folder = self._base_folder / "audios"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._process_and_save_meta()

    def _check_exists(self) -> bool:
        return self._audio_folder.exists() and self._audio_folder.is_dir()

    def _download(self) -> None:
        """Download the dataset if it does not exist already."""
        if self._check_exists():
            return
        for path in self._base_folder.glob("**/*"):
            os.chmod(path, 0o755)
        download_and_extract_archive(self._BVCC_URL[0], download_root=self._base_folder, md5=self._BVCC_URL[1])
        self._gather()

    # copy vcc data to out_folder
    def _gather(self) -> None:
        self._audio_folder.mkdir(exist_ok=True, parents=True)
        for src_wav in (self._base_folder / "main/DATA/wav").glob("*.wav"):
            shutil.copy(src_wav, self._audio_folder / src_wav.name)

    def _process_split(self, split="train") -> None:
        split_to_fname = {"train": "TRAINSET", "eval": "DEVSET", "test": "TESTSET"}

        label_csv_path = self._base_folder / "main/DATA/sets" / split_to_fname[split]
        data_df = pd.read_csv(label_csv_path, header=None)
        data_df.columns = ["system", "file_name", "score", "user_id", "info"]

        grouped = data_df.groupby("file_name")["score"].apply(list).reset_index()
        max_scores = grouped["score"].map(len).max()
        scores_expanded = pd.DataFrame(grouped["score"].to_list(), columns=[f"score_{i+1}" for i in range(max_scores)])

        meta_df = pd.concat([grouped["file_name"], scores_expanded], axis=1)
        meta_df = meta_df[meta_df["file_name"].apply(lambda x: (self._audio_folder / x).is_file())].reset_index(drop=True)
        score_cols = [c for c in meta_df.columns if c.startswith("score_")]
        meta_df["avg_score"] = meta_df[score_cols].mean(axis=1).astype(float)

        cols_order = ["file_name", "avg_score"] + [f"score_{i+1}" for i in range(max_scores)]
        return meta_df[cols_order]

    def _process_and_save_meta(self) -> None:
        train_df = self._process_split("train")
        eval_df = self._process_split("eval")
        test_df = self._process_split("test")

        self._meta_folder.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(self._meta_folder / "train.csv", index=False)
        eval_df.to_csv(self._meta_folder / "eval.csv", index=False)
        test_df.to_csv(self._meta_folder / "test.csv", index=False)


if __name__ == "__main__":
    VCC_NOISE("./data", download=True)
