import matplotlib.pyplot as plt
import pandas as pd


def plot_patient_vs_control(
        patient: pd.DataFrame,
        control: pd.DataFrame,
        show_lines: bool = False
) -> None:
    fig, axs = plt.subplots(figsize=(15, 6), ncols=2, nrows=1, dpi=150)
    if show_lines:
        line_kwargs = {"linestyle": "-.", "lw": 0.5, "alpha": 0.5}
    else:
        line_kwargs = {"linestyle": "-.", "lw": 0.5, "alpha": 0.0}
    _ = patient.plot.scatter(ax=axs[0], x="Fixation point X", y="Fixation point Y", c="Recording timestamp",
                             cmap='viridis', colorbar=False)
    _ = patient.plot(ax=axs[0], x="Fixation point X", y="Fixation point Y", **line_kwargs)
    _ = control.plot.scatter(ax=axs[1], x="Fixation point X", y="Fixation point Y", c="Recording timestamp",
                             cmap='viridis', colorbar=False)
    _ = control.plot(ax=axs[1], x="Fixation point X", y="Fixation point Y", **line_kwargs)
    axs[0].set_title("LCR patient")
    axs[1].set_title("Control patient")
    axs[0].set_aspect("equal")
    axs[1].set_aspect("equal")
    axs[0].get_legend().remove()
    _ = axs[1].get_legend().remove()
    return None
