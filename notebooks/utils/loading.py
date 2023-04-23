import pandas as pd
from pathlib import Path


def load_patient_data(patient_name: str, data_path=Path("/home/pl/MVA/Medic/data/LCR/raw")):
    df = pd.read_excel(data_path / (patient_name + ".xlsx"))
    return df


def load_dataset(raw_path: Path, subset: bool = True, csv: bool = False, **pd_kwargs):
    """
    Load dataset when provided path to raw data
    :param raw_path: Path to patient subgroup (LPR/control)
    :param subset: If True load the subset version (only relevant columns)
    :param csv: If True, load csv files
    :param pd_kwargs: kwargs for pandas read_excel method
    :return: Dictionary {patient_name: dataframe}
    """
    dataset = dict()
    subdir = "subset" if subset else "raw"
    if csv:
        subdir = "subset_csv"
    total = len([_ for _ in (raw_path/subdir).iterdir()])
    for p in tqdm((raw_path/subdir).iterdir(), desc=f"Loading {raw_path.name} dataset", unit=" files", total=total):
        if p.is_file():
            name = p.stem.strip("_subset")
            if csv:
                dataset[name] = pd.read_csv(p, **pd_kwargs)
            else:
                dataset[name] = pd.read_excel(p, **pd_kwargs)
    return dataset
