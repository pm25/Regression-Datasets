import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path


if __name__ == "__main__":
    dataset_dir = Path("./data/utkface")

    train_csv = dataset_dir / "meta" / "train.csv"
    train_df = pd.read_csv(train_csv)
    train_df.rel_path = train_df.rel_path.apply(lambda x: dataset_dir / x)

    imgs = []
    for path in tqdm(train_df.rel_path.tolist()):
        img = Image.open(path).convert("RGB")
        img = np.asarray(img)
        imgs.append(img)

    imgs = np.stack(imgs, axis=0)  # output imgs is a numpy array (n, h, w, c)
    imgs = imgs / 255
    mean = imgs.mean(axis=(0, 1, 2))
    std = imgs.std(axis=(0, 1, 2))
    print(f"Mean : {mean}, STD: {std}")
