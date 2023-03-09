"""Analysis on the succesion fixation/saccade over all sample"""
from typing import Tuple, List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from tqdm import tqdm


def get_sf_df(path_to_dataset: Path, **read_kwargs) -> pd.DataFrame:
    """
    Get the eye movement type Dataframe with the timestamp. We remove unlabeled movement type
    :param path_to_dataset:
    :return: dataframe
    """
    df = pd.read_excel(path_to_dataset, **read_kwargs)
    df = df.loc[df["Eye movement type"].isin(["Saccade", "Fixation"])]
    return df


def get_sf_list(df: pd.DataFrame) -> List[Tuple[str, float]]:
    """
    Get the list [mov_type, duration] corresponding to the respective duration of each movement type in their
    succession order.
    :param df: Dataframe with rows ["Recording timestamp","Eye movement type"]
    :return: list [mov_type, duration]
    """
    start_type = df.iloc[0, 1]
    timestamp_0 = df.iloc[0, 0]
    duration = 0
    sf_list = []
    for timestamp, type_movement in df.values:
        if type_movement == start_type:
            continue
        else:
            duration = timestamp - timestamp_0
            sf_list.append((type_movement, duration))
            timestamp_0 = timestamp
            duration = 0
            start_type = type_movement
    return sf_list


def get_data(key: str, sf_list: List[Tuple]) -> np.array:
    return np.asarray([item[1] for item in sf_list if item[0] == key])


def get_all_data(key: str, sf_list_all: List[List[Tuple]]) -> List[np.array]:
    all_data = []
    for data in sf_list_all:
        all_data.append(get_data(key, data))
    return all_data


def plot_sf_list(sf_list: List, fig_title: str = None) -> None:
    """Plot the sf list data in bar plot"""

    saccade = [(i, sf[1]) for i, sf in enumerate(sf_list) if sf[0] == "Saccade"]
    fixation = [(i, sf[1]) for i, sf in enumerate(sf_list) if sf[0] == "Fixation"]

    plt.figure(figsize=(15, 5), dpi=120)

    plt.bar(
        [s[0] for s in saccade],
        [s[1] for s in saccade],
        label="Saccade",
        color='g'
    )

    plt.bar(
        [f[0] for f in fixation],
        [f[1] for f in fixation],
        label="Fixation",
        color='orange'
    )
    plt.xticks()
    plt.legend()
    if fig_title is not None:
        plt.title(fig_title)
    _ = plt.show()


def make_stats(sf_data_patient: List[List], sf_data_control: List[List], save=True, normalize=False,
               truncate=True) -> None:
    """
    Make statistics on the data over the repartition of duration of saccade and fixation
    :param sf_data_patient: List of sf_list (str,float) for patient
    :param sf_data_control: List of sf_list (str,float) for control group
    :param save: Save figure if True
    :param normalize: Standardize data if True
    :param truncate:Truncate over 3*sigma for plotting
    :return:
    """
    fix_ct, fix_pa, sac_ct, sac_pa = list(), list(), list(), list()

    for sf_pa, sf_ct in zip(sf_data_patient, sf_data_control):
        fix_ct += [item[1] for item in sf_ct if item[0] == "Fixation"]
        fix_pa += [item[1] for item in sf_pa if item[0] == "Fixation"]
        sac_ct += [item[1] for item in sf_ct if item[0] == "Saccade"]
        sac_pa += [item[1] for item in sf_pa if item[0] == "Saccade"]
    data = [fix_ct, fix_pa, fix_pa + fix_ct, sac_ct, sac_pa, sac_pa + sac_ct]
    data = [np.asarray(d) for d in data]
    if normalize:
        data = [(d - np.min(d)) / (np.max(d) - np.min(d)) for d in data]
    titles = ["Contrôle", "LPR", "Both", "", "", ""]
    labels_y = ["Fixations duration", "", "", "Saccades duration", "", ""]
    labels_x = [""] * 3 + ["Time (s)"] * 3
    fig = plt.figure(figsize=(15, 10), dpi=150)
    for i in range(6):
        sigma = np.std(data[i])
        mean = np.mean(data[i])
        if truncate:
            data[i] = data[i][np.where(data[i] < 4 * sigma)]
        plt.subplot(2, 3, i + 1)
        plt.hist(data[i], bins=50)
        plt.gca().get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: x / 1e6))
        plt.xlabel(labels_x[i])
        plt.ylabel(labels_y[i], fontsize=14)
        plt.axvline(sigma, label=r'$\sigma$ : ' + f"{np.mean(data[i] < sigma) * 100:.2f}\%", color='red',
                    linestyle="--")
        plt.axvline(2 * sigma, label=r'$2\sigma$ : ' + f"{np.mean(data[i] < 2 * sigma) * 100:.2f}\%", color='orange',
                    linestyle="--")
        plt.axvline(3 * sigma, label=r'$3\sigma$ : ' + f"{np.mean(data[i] < 3 * sigma) * 100:.2f}\%", color='green',
                    linestyle="--")
        plt.title(titles[i], fontsize=14)
        plt.legend()
    if save:
        plt.savefig("Stats.pdf")
    plt.show()


def read_all_data(path_to_data: Path, **read_kwargs) -> Tuple[List[str], List[List]]:
    """
    Read all data and return the saccade fixation list
    :param path_to_data:
    :return:
    """
    names = list()
    sf_list_all = list()
    tot = len([_ for _ in path_to_data.iterdir()])
    for p in tqdm(path_to_data.iterdir(), desc=f"Loading dataset {path_to_data}", total=tot, unit="Files"):
        df = get_sf_df(p, **read_kwargs)
        sf_list = get_sf_list(df)
        names.append(p.stem.strip("_subset"))
        sf_list_all.append(sf_list)
    return names, sf_list_all


def get_symbols_sequence(symbols_saccade, symbols_fixation, sf_list_all) -> List[List[str]]:
    """
    Get the whole sequence of symbols given the respective sequence for saccade and fixation
    :param symbols_saccade: List of each saccade symbols sequence
    :param symbols_fixation: Same for fixation
    :param sf_list_all: Original data (as returned by read_all_data)
    :return:
    """
    sequence_symbols = []
    for sym_s, sym_f, sf_all in zip(symbols_saccade, symbols_fixation, sf_list_all):
        res = np.empty(len(sf_all), dtype=str)
        idx = np.asarray([0 if item[0] == "Saccade" else 1 for item in sf_all])
        res[np.where(idx==0)] = sym_s
        res[np.where(idx==1)] = sym_f
        sequence_symbols.append(res)
    return sequence_symbols
