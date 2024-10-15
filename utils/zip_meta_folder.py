# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import os
from pathlib import Path


if __name__ == "__main__":
    dataset_dir = Path("./data/utkface")
    meta_path = dataset_dir / "meta.zip"

    if meta_path.is_file():
        meta_path.unlink()

    os.system(f"cd {dataset_dir} && zip -r meta.zip meta && cd ..")
