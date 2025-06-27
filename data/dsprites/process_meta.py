# Code in this file is adapted from pytorch/pytorch
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

from torchvision.datasets.utils import download_url


class DSprites:
    """`dSprites Dataset <https://github.com/google-deepmind/dsprites-dataset>`
    Args:
        root (string): Root directory of the dataset.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "https://raw.githubusercontent.com/google-deepmind/dsprites-dataset/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    _MD5 = "7da33b31b13a06f4b04a70402ce90c2e"

    def __init__(
        self,
        root: str,
        download: bool = False,
    ) -> None:
        self._base_folder = Path(root) / "dsprites"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"
        self._data_file = self._base_folder / Path(self._URL).name

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
        download_url(self._URL, root=self._base_folder, md5=self._MD5)
        self._extract_heart_images()

    def _extract_heart_images(self) -> None:
        """Extracts only the 'heart' shape images and saves them as PNGs with metadata."""
        dataset = np.load(self._data_file, allow_pickle=True, encoding="latin1")
        images = dataset["imgs"]
        latents_values = dataset["latents_values"]

        self._images_folder.mkdir(parents=True, exist_ok=True)
        records = []

        idx = 0
        assert len(images) == len(latents_values)

        for image, latents in tqdm(zip(images, latents_values), total=len(images), dynamic_ncols=True):
            shape = latents[1]
            if shape != 3:  # Only include heart-shaped objects
                continue

            filename = f"{idx:06d}.png"
            idx += 1

            image_uint8 = (image * 255).astype("uint8")
            Image.fromarray(image_uint8, mode="L").save(self._images_folder / filename)

            records.append(
                {
                    "file_name": filename,
                    "scale": latents[2],
                    "orientation": latents[3],
                    "x_position": latents[4],
                    "y_position": latents[5],
                }
            )

        self._meta_folder.mkdir(parents=True, exist_ok=True)
        records_df = pd.DataFrame(records)
        records_df = records_df[~np.isclose(records_df["orientation"], 2 * np.pi)]
        records_df.to_csv(self._meta_folder / "raw_meta.csv", index=False)

    def _process_and_save_meta(self) -> None:
        raw_meta_df = pd.read_csv(self._meta_folder / "raw_meta.csv")
        meta_df = raw_meta_df[["file_name", "orientation"]].rename(columns={"orientation": "label"})

        meta_df.file_name = meta_df.file_name.apply(lambda x: Path(x).name)
        meta_df = meta_df.sort_values(by=["file_name"])
        meta_df.label = meta_df.label.astype(float)
        test_df = meta_df.sample(n=10000, random_state=222)
        train_df = meta_df.drop(test_df.index)

        self._meta_folder.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(self._meta_folder / "train.csv", index=False)
        test_df.to_csv(self._meta_folder / "test.csv", index=False)


if __name__ == "__main__":
    DSprites("./data", download=True)
