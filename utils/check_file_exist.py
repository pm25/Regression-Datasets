import pandas as pd
from tqdm import tqdm
from pathlib import Path


if __name__ == "__main__":
    dataset_dir = Path("./data/utkface")

    train_df = pd.read_csv(dataset_dir / "meta" / "train.csv")
    for data_rel_path in tqdm(train_df.rel_path.tolist()):
        data_path = dataset_dir / data_rel_path
        assert data_path.is_file(), f"can't find {data_path}"

    test_df = pd.read_csv(dataset_dir / "meta" / "test.csv")
    for data_rel_path in tqdm(test_df.rel_path.tolist()):
        data_path = dataset_dir / data_rel_path
        assert data_path.is_file(), f"can't find {data_path}"

    print("Check completed, all training and testing data exist!")
