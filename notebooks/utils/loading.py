import pandas as pd
from pathlib import Path


def load_patient_data(patient_name: str, data_path=Path("/home/pl/MVA/Medic/data/LCR/raw")):
    df = pd.read_excel(data_path / (patient_name + ".xlsx"))
    return df
