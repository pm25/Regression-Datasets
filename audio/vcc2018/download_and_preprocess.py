# modify from: https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

import io
import pandas as pd
from pathlib import Path

from torchvision.datasets.utils import download_and_extract_archive


class VCC2018:
    """`The VCC2018 Data Set <https://datashare.ed.ac.uk/handle/10283/3061><https://datashare.ed.ac.uk/handle/10283/3257>`_.
    Args:
        root (string): Root directory of the dataset.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _DATA_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_submitted_systems_converted_speech.tar.gz"
    _DATA_MD5 = "75b0f937240f6850a56ec2cbad34b4ad"
    _LABEL_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3257/vcc2018_listening_test_scores.zip"
    _LABEL_MD5 = "aad7e0ce99279a6f16e8a0d6f963c1c9"

    def __init__(
        self,
        root: str,
        download: bool = False,
    ) -> None:
        self._base_folder = Path(root) / "vcc2018"
        self._meta_folder = self._base_folder / "meta"
        self._audio_folder = self._base_folder / "mnt/sysope/test_files/testVCC2"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._process_and_save_meta()

    def _check_exists(self) -> bool:
        return self._audio_folder.exists() and self._audio_folder.is_dir()

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._LABEL_URL, download_root=self._base_folder, md5=self._LABEL_MD5)
        download_and_extract_archive(self._DATA_URL, download_root=self._base_folder, md5=self._DATA_MD5)

    def _process_and_save_meta(self) -> None:
        label_csv_path = self._base_folder / "vcc2018_listening_test_scores" / "vcc2018_evaluation_mos_simple.txt"
        with open(label_csv_path, "r") as f:
            data = f.read().replace(",\n", "\n")
            data_df = pd.read_csv(io.StringIO(data))

        data_df = data_df[["SYSTEM_TARGET-SPEAKER_SOURCE-SPEAKER_SENTENCE_TASK", "SCORE"]].rename(
            columns={"SYSTEM_TARGET-SPEAKER_SOURCE-SPEAKER_SENTENCE_TASK": "file_name", "SCORE": "label"}
        )
        meta_df = data_df.groupby("file_name", as_index=False).mean()

        meta_df = meta_df.sort_values(by=["file_name"])
        meta_df.label = meta_df.label.astype(float)
        train_df = meta_df.sample(frac=0.8, random_state=222)
        test_df = meta_df.drop(train_df.index)

        self._meta_folder.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(self._meta_folder / "train.csv", index=False)
        test_df.to_csv(self._meta_folder / "test.csv", index=False)


if __name__ == "__main__":
    VCC2018("./audio", download=True)
