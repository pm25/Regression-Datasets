# Code in this file is adapted from pytorch/pytorch
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

from pathlib import Path

from torchvision.datasets.utils import download_url


class AMAZON_REVIEW:
    """`Amazon Review Dataset`.
    Args:
        root (string): Root directory of the dataset.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL_MD5 = [
        ("https://huggingface.co/datasets/py97/Amazon-Review/resolve/main/train.json", "304672e9ed4696f035cce447f5b8bab4"),
        ("https://huggingface.co/datasets/py97/Amazon-Review/resolve/main/dev.json", "08869c3c0473a1c6f134adb547da192f"),
        ("https://huggingface.co/datasets/py97/Amazon-Review/resolve/main/test.json", "8a8c3980256735cdd89de976169f2149"),
    ]

    def __init__(
        self,
        root: str,
        download: bool = False,
    ) -> None:
        self._base_folder = Path(root) / "amazon_review"
        self._data_folder = self._base_folder / "data"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

    def _check_exists(self) -> bool:
        return all([(self._data_folder / file_name).is_file() for file_name in ["train.json", "dev.json", "test.json"]])

    def _download(self) -> None:
        if self._check_exists():
            return
        for url, md5 in self._URL_MD5:
            download_url(url, root=self._data_folder, md5=md5)


if __name__ == "__main__":
    AMAZON_REVIEW("./text", download=True)
