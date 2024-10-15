# Data Folder

In this section, each dataset folder contains the train-test splits and metadata (labels and possibly other information) associated with the datasets. The file structure is as follows:

```
[datasets]
 ├── meta.zip
 ├── meta
 │   ├── raw_meta.csv (optional)
 │   ├── train.csv
 │   ├── dev.csv (optional)
 │   └── test.csv
 └── process_meta.py
```

The `meta.zip` file contains the packed metadata from the `[datasets]/meta` folder. Additionally, each dataset folder includes a `process_meta.py` file, which details how the metadata was processed.
