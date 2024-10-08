from typing import Any

import numpy as np
from pybaselines import Baseline

__all__ = [
    "baseline",
]


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
