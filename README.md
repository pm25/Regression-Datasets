<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->

<a id="readme-top"></a>

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<h1 align="center">🎛️ Regression Datasets</h1>

<!-- TABLE OF CONTENTS -->
<details>
  <summary><strong>📋 Table of Contents</strong></summary>
  <ol>
    <li><a href="#1-installation">Installation</a></li>
    <li><a href="#2-usage">Usage</a></li>
    <li><a href="#3-datasets">Datasets</a></li>
    <li><a href="#4-license">License</a></li>
    <li><a href="#5-contact">Contact</a></li>
    <li><a href="#6-acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

This repository offers a diverse collection of regression datasets across vision, audio and text domains. It provides dataset classes that follow the <a href="https://github.com/pytorch/vision/tree/main/torchvision/datasets">PyTorch Datasets</a> structure, allowing users to automatically download and load these datasets with ease. All datasets come with a permissive license, permitting their use for research purposes.

<!-- Installation -->

## 1. Installation

To install the `regsets` package, you can use pip:

```sh
python -m pip install regsets
```

Alternatively, you can download a specific dataset file (e.g., `utkface.py`) and include it in your project to load the dataset locally.

<!-- USAGE -->

## 2. Usage

Below are examples of how to use the `regsets` package for loading datasets.

### 📸 Vision Datasets

```python
from regsets.vision import UTKFace

utkface_trainset = UTKFace(root="./data", split="train", download=True)

for image, label in utkface_trainset:
    ...
```

### 🎧 Audio Datasets

```python
from regsets.audio import VCC2018

vcc2018_trainset = VCC2018(root="./data", split="train", download=True)

for audio, sample_rate, label in vcc2018_trainset:
    ...
```

### 📝 Text Datasets

```python
from regsets.text import Amazon_Review

amazon_review_trainset = Amazon_Review(root="./data", split="train", download=True)

for texts, label in amazon_review_trainset:
    (ori, aug_0, aug_1) = texts
    ...
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DATASETS -->

## 3. Datasets

For datasets that do not provide a predefined train-test split, I randomly sample 80% of the data for training and reserve the remaining 20% for testing. Details for each dataset are provided below.

### 📸 Vision Datasets

| Dataset | # Training Data | # Dev Data | # Test Data | Target Range |
| ------- | --------------- | ---------- | ----------- | ------------ |
| UTKFace | 18,964          | -          | 4,741       | [1, 116]     |

### 🎧 Audio Datasets

| Dataset | # Training Data | # Dev Data | # Test Data | Target Range |
| ------- | --------------- | ---------- | ----------- | ------------ |
| BVCC    | 4,974           | 1,066      | 1,066       | [1, 5]       |
| VCC2018 | 16,464          | -          | 4,116       | [1, 5]       |

### 📝 Text Datasets

| Dataset       | # Training Data | # Dev Data | # Test Data | Target Range |
| ------------- | --------------- | ---------- | ----------- | ------------ |
| Amazon Review | 250,000         | 25,000     | 650,000     | [0, 4]       |
| Yelp Review   | 250,000         | 25,000     | 50,000      | [0, 4]       |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## 4. License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## 5. Contact

-   Pin-Yen Huang (pyhuang97@gmail.com)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## 6. Acknowledgments

-   [PyTorch](https://github.com/pytorch)
-   [UTKFace](https://susanqq.github.io/UTKFace)
-   [VCC2018](https://datashare.ed.ac.uk/handle/10283/3061)
-   [BVCC](https://zenodo.org/records/6572573)
-   [USB](https://github.com/microsoft/semi-supervised-learning)
-   [Amazon Review](https://dl.acm.org/doi/10.1145/2507157.2507163)
-   [Yelp Review](http://www.yelp.com/dataset_challenge)
-   [README Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
