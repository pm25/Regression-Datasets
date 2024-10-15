# Code in this file is adapted from pytorch/pytorch
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

from pathlib import Path

from torchvision.datasets.utils import download_url


class YELP_REVIEW:
    """`The Yelp Review Dataset`.
    Args:
        root (string): Root directory of the dataset.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL_MD5 = [
        ("https://huggingface.co/datasets/py97/Yelp-Review/resolve/main/train.json", "7fb2af6453cbc781ace81ca121d28567"),
        ("https://huggingface.co/datasets/py97/Yelp-Review/resolve/main/dev.json", "6d4850b0e87eaf624e76c4e50e059a4b"),
        ("https://huggingface.co/datasets/py97/Yelp-Review/resolve/main/test.json", "b4fbff480ecb60208dad8e5e08572853"),
    ]

    def __init__(
        self,
        root: str,
        download: bool = False,
    ) -> None:
        self._base_folder = Path(root) / "yelp_review"
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
    YELP_REVIEW("./text", download=True)
