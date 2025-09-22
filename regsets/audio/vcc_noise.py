# Code in this file is adapted from pytorch/pytorch
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

import shutil
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Tuple

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg


class VCC_NOISE(Dataset):
    """`The VCC2018 Data Set <https://datashare.ed.ac.uk/handle/10283/3061> <https://datashare.ed.ac.uk/handle/10283/3257>`

    The Voice Conversion Challenge 2018 (VCC2018) dataset is an audio quality assessment dataset,
    where the objective is to predict the quality of an audio sample. The labels, ranging from 1
    to 5, are obtained by averaging the scores provided by multiple listeners. The dataset
    comprises over 20,000 audio files, which we split into 16,464 training samples and 4,116 test samples.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL_MD5 = {
        "data": ("https://zenodo.org/records/10691660/files/main.tar.gz", "ba4b896801282ca0eae37b9fd81ed94c"),
        "meta": ("https://github.com/pm25/regression-datasets/raw/refs/heads/main/data/vcc_noise/meta.zip", "889d4bd22c706fdd4bcc8a67b0808172"),
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = False,
        num_judges: int = -1,  # number of judges to sample, -1 means use all
        seed: int = 22,  # random seed for deterministic sampling
    ) -> None:
        super().__init__()
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(root) / "vcc_noise"
        self._meta_folder = self._base_folder / "meta"
        self._audio_folder = self._base_folder / "audios"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        metadata = pd.read_csv(self._meta_folder / f"{split}.csv")
        self._file_paths = metadata["file_name"].apply(lambda x: self._audio_folder / x).to_numpy(dtype="object")

        score_cols = [c for c in metadata.columns if c.startswith("score_")]
        scores_array = metadata[score_cols].to_numpy(dtype=np.float32)

        if num_judges == -1 or num_judges >= scores_array.shape[1]:
            self._labels = scores_array.mean(axis=1)
        else:
            rng = np.random.default_rng(seed)
            self._labels = np.array([np.mean(rng.choice(row, size=num_judges, replace=False)) for row in scores_array], dtype=np.float32)

    def __len__(self) -> int:
        return len(self._file_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        audio_file, label = self._file_paths[idx], self._labels[idx]
        waveform, sample_rate = librosa.load(audio_file, sr=None, mono=True)
        return waveform, sample_rate, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._audio_folder))

    def _download(self) -> None:
        if self._check_exists():
            return
        for url, md5 in self._URL_MD5.values():
            download_and_extract_archive(url, download_root=self._base_folder, md5=md5)
        self._gather()

    def _gather(self) -> None:
        self._audio_folder.mkdir(exist_ok=True, parents=True)
        for src_wav in (self._base_folder / "main/DATA/wav").glob("*.wav"):
            shutil.copy(src_wav, self._audio_folder / src_wav.name)
