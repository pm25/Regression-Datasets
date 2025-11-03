import re
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET

from torchvision.datasets.utils import extract_archive

# -------------------------------------------------------------------------
# XML Parsing Helpers
# -------------------------------------------------------------------------
def strip_ns(tag: str) -> str:
    """Remove XML namespace from tag."""
    return tag.split("}")[-1] if "}" in tag else tag


def element_to_dict(elem: ET.Element) -> dict:
    """Convert an XML Element recursively to a Python dictionary."""
    node = {**elem.attrib}

    children = list(elem)
    if children:
        for child in children:
            child_tag = strip_ns(child.tag)
            child_val = element_to_dict(child)
            node.setdefault(child_tag, []).append(child_val)
        # Simplify single-item lists
        for k, v in list(node.items()):
            if isinstance(v, list) and len(v) == 1:
                node[k] = v[0]
    elif elem.text and elem.text.strip():
        return elem.text.strip()

    return node


def xml_to_dict(xml_file_path: Path) -> dict:
    """Parse an XML file into a dictionary."""
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    return {strip_ns(root.tag): element_to_dict(root)}


# -------------------------------------------------------------------------
# ADNI Dataset
# -------------------------------------------------------------------------
class ADNI:
    """
    ADNI Dataset (T1-weighted MR Images) <https://adni.loni.usc.edu>.

    This class handles preprocessing for the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset.
    The dataset is not publicly downloadable; users must manually obtain the files from the
    official ADNI website: https://adni.loni.usc.edu

    To prepare the dataset, download the following four archives and place them in ``data/adni``:

    - ``ADNI1_Complete 1Yr 1.5T.zip``
    - ``ADNI1_Complete_1Yr_1.5T_metadata.zip``
    - ``ADNI1_Complete 1Yr 3T.zip``
    - ``ADNI1_Complete_1Yr_3T_metadata.zip``

    Once the files are placed in the correct directory, executing this script will automatically
    extract the archives, process the XML metadata, and generate the dataset splits
    (`train.csv`, `eval.csv`) under ``data/adni/meta``.

    By default, only cognitively normal (CN) subjects are used as the validation data.

    Parameters
    ----------
    root : str
        Root directory where the dataset is stored.
    download : bool, optional
        If ``True``, attempts to download the dataset automatically (not supported for ADNI).
        Default is ``False``.

    Notes
    -----
    - The dataset must be downloaded manually due to ADNI's data access restrictions.
    - The preprocessing automatically selects the earliest available MRI scan per subject (no duplicate subject across dataset).
    """

    _data_files = [
        "ADNI1_Complete 1Yr 1.5T.zip",
        "ADNI1_Complete_1Yr_1.5T_metadata.zip",
        "ADNI1_Complete 1Yr 3T.zip",
        "ADNI1_Complete_1Yr_3T_metadata.zip",
    ]

    def __init__(self, root: str, download: bool = False) -> None:
        self._base_folder = Path(root) / "adni"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "t1-weighted"
        self._extracted_folder = self._base_folder / "ADNI"

        if download:
            self._download()

        if not self._check_extracted():
            self._extract()

        self._process_and_save_meta()

    def _check_exists(self) -> bool:
        return all((self._base_folder / f).exists() for f in self._data_files)

    def _check_extracted(self) -> bool:
        return self._extracted_folder.exists()

    def _download(self) -> None:
        """ADNI data must be manually downloaded."""
        if self._check_exists():
            return
        raise RuntimeError(
            "ADNI is not publicly downloadable. "
            f"Please download the files ({self._data_files}) manually and place them in {self._base_folder}."
        )

    def _extract(self) -> None:
        """Extract archives."""
        for file_name in self._data_files:
            archive = self._base_folder / file_name
            print(f"Extracting {archive.name} ...")
            extract_archive(archive, self._base_folder, remove_finished=False)

    # -------------------------------------------------------------------------
    # Data processing
    # -------------------------------------------------------------------------
    _xml_pattern = re.compile(r"ADNI_(\d+_S_\d+)_.*_S(\d+)_I(\d+)\.xml")

    def locate_nii_from_xml(self, xml_file: Path) -> Path:
        """Find the corresponding NIfTI file given an XML file."""
        match = self._xml_pattern.match(xml_file.name)
        if not match:
            return None

        subject_id, series_id, image_id = match.groups()
        subject_folder = self._extracted_folder / subject_id / "MPR__GradWarp__B1_Correction__N3__Scaled"

        if not subject_folder.exists():
            print(f"[Warning] Missing subject folder for {subject_id}")
            return None

        for nii_path in subject_folder.rglob(f"I{image_id}/*.nii"):
            if nii_path.name.endswith(f"S{series_id}_I{image_id}.nii"):
                return nii_path
        return None

    def get_earliest_file_per_subject(self):
        """Select the earliest scan (smallest series ID) per subject."""
        earliest_files = {}
        pattern = re.compile(r"ADNI_(\d+_S_\d+)_.*_S(\d+)_I\d+\.xml")

        for xml_path in self._extracted_folder.glob("*.xml"):
            if not xml_path.is_file():
                continue
            if "MPR__GradWarp__B1_Correction" not in xml_path.name or "Scaled_2_" in xml_path.name:
                continue

            match = pattern.match(xml_path.name)
            if match:
                subject_id, series_id = match.group(1), int(match.group(2))
                prev = earliest_files.get(subject_id)
                if prev is None or series_id < prev[0]:
                    earliest_files[subject_id] = (series_id, xml_path)

        selected_files = [tup[1] for tup in earliest_files.values()]
        print(f"✅ Selected {len(selected_files)} earliest scan XML files.")
        return selected_files

    def _process_and_save_meta(self) -> None:
        """Process XML metadata and create train/validation splits."""
        selected_files = self.get_earliest_file_per_subject()
        processed_metadata = []

        self._images_folder.mkdir(exist_ok=True, parents=True)

        for xml_path in tqdm(selected_files, desc="Processing metadata"):
            metadata = xml_to_dict(xml_path)
            project = metadata.get("idaxs", {}).get("project", {})
            subject = project.get("subject", {})

            visit_identifier = (
                subject.get("visit", {}).get("visitIdentifier")
            )
            age_at_scan = subject.get("study", {}).get("subjectAge")
            research_group = subject.get("researchGroup")

            nii_path = self.locate_nii_from_xml(xml_path)
            if nii_path is None:
                print(f"[Warning] No NIfTI found for {xml_path.name}")
                continue

            shutil.copy2(nii_path, self._images_folder)
            processed_metadata.append(
                {
                    "image_name": nii_path.name,
                    "visit_identifier": visit_identifier,
                    "age_at_scan": float(age_at_scan) if age_at_scan else None,
                    "research_group": research_group,
                }
            )

        raw_meta_df = pd.DataFrame(processed_metadata)
        meta_df = raw_meta_df[["image_name", "age_at_scan", "research_group"]].rename(columns={"image_name": "file_name", "age_at_scan": "label"})
        
        cn_df = meta_df[meta_df["research_group"] == "CN"]
        non_cn_df = meta_df[meta_df["research_group"] != "CN"]

        # === split ===
        train_cn_df = cn_df.sample(frac=0.5, random_state=222)
        val_df = cn_df.drop(train_cn_df.index)
        
        train_df = pd.concat([non_cn_df, train_cn_df], ignore_index=True)

        self._meta_folder.mkdir(exist_ok=True, parents=True)
        raw_meta_df.to_csv(self._meta_folder / "raw_meta.csv", index=False)
        train_df.to_csv(self._meta_folder / "train.csv", index=False)
        val_df.to_csv(self._meta_folder / "val.csv", index=False)


if __name__ == "__main__":
    ADNI("./data", download=False)
