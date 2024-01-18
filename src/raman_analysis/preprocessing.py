from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pybaselines import Baseline
from scipy.signal import find_peaks

__all__ = [
    "find_cosmic_rays",
    "remove_cosmic_rays",
    "group_spectra_points",
    "baseline",
]


def find_cosmic_rays(
    spectra: np.ndarray, ignore_region: tuple[int, int] = (200, 400), **kwargs: Any
) -> np.ndarray:
    """
    Find the indices of cosmic rays.

    Parameters
    ----------
    spectra : (N, 1340)
        The arraylike of spectr to search through
    ignore_region : (int, int)
        A region to not worry about cosmic rays in.
    **kwargs :
        Passed to scipy.signal.find_peaks

    Returns
    -------
    idx : (M, 2) np.ndarray
        The indices of which spectra+pixels have detected cosmic rays.
        The first column contains which spectra, the second which pixel.
    """
    spectra = np.atleast_2d(spectra)
    idx = []
    min_ignore = min(ignore_region)
    max_ignore = max(ignore_region)
    threshold = kwargs.pop("threshold", 75)
    prominence = kwargs.pop("prominence", 100)
    for s, spec in enumerate(spectra):
        peaks, _ = find_peaks(
            spec, threshold=threshold, prominence=prominence, **kwargs
        )
        for p in peaks:
            if min_ignore < p < max_ignore:
                continue
            idx.append((s, p))
    return np.asarray(idx)


def remove_cosmic_rays(
    df: pd.DataFrame, plot: bool = False, **kwargs: Any
) -> pd.DataFrame:
    """
    Process a dataframe by removing all spectra with detected cosmic rays.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe with spectra components as the first 1340 columns
    plot : bool
        Whether to generate a plot showing which spectra were removed.
    **kwargs
        Passed to scipy.signal.find_peaks

    Returns
    -------
    pd.DataFrame
        The spectra dataframe with spectra with cosmic rays removed.
    """
    spectra = df.iloc[:, :1340].values

    cosmic_idx = find_cosmic_rays(spectra, **kwargs)
    keep_idx = np.setdiff1d(np.arange(spectra.shape[0]), cosmic_idx)
    if plot:
        import matplotlib.pyplot as plt

        unique_cosmic, offset_idx = np.unique(cosmic_idx[:, 0], return_inverse=True)
        offsets = np.arange(unique_cosmic.shape[0]) * 100
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        axs[0].set_title("Post Removal")
        axs[0].plot(spectra[keep_idx].T, alpha=0.75)

        axs[1].plot(spectra[unique_cosmic].T + offsets)
        axs[1].plot(
            cosmic_idx[:, 1],
            spectra[cosmic_idx[:, 0], cosmic_idx[:, 1]] + offsets[offset_idx],
            "rx",
            markersize=10,
            label="detected cosmic rays",
            mew=5,
        )
        axs[1].legend()

        axs[1].set_title("Post removal - with offset")
    return df.iloc[keep_idx]


def group_spectra_points(df: pd.DataFrame, multiplier: int) -> pd.DataFrame:
    """
    Add which point each spectra is from to the multiindex.

    Parameters
    ----------
    df : pd.DataFrame
        The raman dataframe. Should be organized as T, P, type
    multiplier : int
        How many subspectra per point

    Returns
    -------
    pd.DataFrame
        the original dataframe with pt_label appended to multiindex
    """
    offset = 0
    for pos, pos_df in df.groupby(level=1):
        for t, tp_df in pos_df.groupby(level=0):
            n_pts = int(len(tp_df) / multiplier)
            pt_labels = (
                np.broadcast_to(
                    np.arange(n_pts, dtype=int)[:, None], (n_pts, multiplier)
                ).ravel()
                + offset
            )
            df.loc[(t, pos), "pt"] = pt_labels
        offset = df.loc[pos, "pt"].max()
    df["pt"] = df["pt"].astype(int)
    return df.set_index("pt", append=True)


def baseline(spectra: np.ndarray, method: str = "arpls", **params: Any) -> np.ndarray:
    """
    Calculate the baseline of [many] spectra using pybaselines.

    Parameters
    ----------
    spectra : array-like ([N], wns)
        The spectra to calculate the baseline of.
    method : str, default: "arpls"
        The pybaselines method name.
    **params:
        Passed to pybaselines

    Returns
    -------
    baseline : np.ndarray ([N], wns)
        The calculated baselines
    """
    baseliner = Baseline(np.arange(spectra.shape[-1]))
    baseline_func = getattr(baseliner, method)

    spectra = np.atleast_2d(spectra)
    if np.issubdtype(spectra.dtype, np.integer):
        spectra = spectra.astype(np.float32)

    baselines = np.zeros_like(spectra)

    for i, spec in enumerate(spectra):
        baselines[i], w = baseline_func(spec, **params)
    return baselines.squeeze()
