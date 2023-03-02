import pandas as pd
from pathlib import Path

from tqdm import tqdm


def load_patient_data(patient_name: str, data_path: Path = Path("/home/pl/MVA/Medic/data/LCR/raw"), **pd_kwargs):
    df = pd.read_excel(data_path / (patient_name + ".xlsx"), **pd_kwargs)
    return df


def load_dataset(raw_path: Path, subset: bool = True, **pd_kwargs) -> dict[str, pd.DataFrame]:
    """
    Load dataset when provided path to raw data
    :param raw_path: Path to patient subgroup (LPR/control)
    :param subset: If True load the subset version (only relevant columns)
    :param pd_kwargs: kwargs for pandas read_excel method
    :return: Dictionary {patient_name: dataframe}
    """
    dataset = dict()
    subdir = "subset" if subset else "raw"
    total = len([_ for _ in (raw_path/subdir).iterdir()])
    for p in tqdm((raw_path/subdir).iterdir(), desc=f"Loading {raw_path.name} dataset", unit=" files", total=total):
        if p.is_file():
            name = p.stem.strip("_subset")
            dataset[name] = pd.read_excel(p, **pd_kwargs)
    return dataset
