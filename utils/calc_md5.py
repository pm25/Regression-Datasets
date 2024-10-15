from pathlib import Path
from torchvision.datasets.utils import calculate_md5

dataset_dir = "./data/yelp_review"
data_exts = [".tar", ".gz", ".zip"]

for path in Path(dataset_dir).glob("*"):
    if path.suffix in data_exts:
        md5 = calculate_md5(path)
        print(f"{path.name}: {md5}")
