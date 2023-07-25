from typing import Iterable, Union

import numpy as np
import pandas as pd
import xarray as xr


def ds2df(ds: xr.Dataset, fov: int, cell_index_start: int = 0, filename=None) -> pd.DataFrame:
    """
    Convert a single dataset into a dataframe.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to convert
    fov : int
        The fov index.
    cell_index_start : int
        The offset of the cell index. This is necessary
        if you want to easily combine multiple datasets into a
        single dataframe down the road
    filename : str, optional
        The filename to add to an 'fname' column.

    Returns
    -------
    df : dataframe
    """
    pts_cell = ds.pts_cell
    mult = ds.attrs["multiplier"]
    indx = pd.MultiIndex.from_product(
        (np.arange(int(pts_cell.shape[0] / mult)) + cell_index_start, np.arange(mult)),
        names=("cell", "sub-cell"),
    )
    df = pd.DataFrame(ds["cell_raman"][0], index=indx)
    df["x"] = ds["cell_points"][:, 0]
    df["y"] = ds["cell_points"][:, 1]

    cell_points = ds["cell_points"] * 2048
    bkd_points = ds["bkd_points"] * 2048
    cell_raman = ds["cell_raman"] - 608
    bkd_raman = ds["bkd_raman"] - 608

    thres = 140

    cell_com = cell_points.to_numpy().astype(int)
    gfp_int = np.asarray([ds["img"][1, 0, x[0], x[1]].values for x in cell_com])

    df["gfp_int"] = gfp_int
    df["fov"] = fov
    if filename is not None:
        df["fname"] = filename
    return df




def glob2df(
    files: Union[Iterable[str], str],
    conditions,
    threshold: float,
    well_number: int = 0,
    cell_index_start: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    files : iterable, str
        An iterable of filenames, or a glob string to use to find the files.
    conditions : tuple(str, str)
        The two conditions in this dataset, should be listed
        in order of (undyed condition, dyed condition)
    threshold : float
        The threshold gfp value for determining dyed vs undyed cell
    well_number : int, default 0
        What well number to put in the dataframe
    cell_index_start : int
        The offset of the cell index. This is necessary
        if you want to easily combine multiple datasets into a
        single dataframe down the road
    verbose : bool, default True
        Whether to print out the file names as they are opened.

    Returns
    -------
    df : pd.DataFrame
    images : xr.DataArray
    """
    if isinstance(files, str):
        files = glob(files)
    dfs = []
    images = []
    for fov, f in enumerate(files):
        if verbose:
            print(f)
        ds = xr.open_dataset(f)
        dfs.append(ds2df(ds, fov, cell_index_start, filename=f))
        cell_index_start = dfs[-1].index.max()[0] + 1
        images.append(ds["img"])
    images = xr.concat(images, dim="fov")
    df = pd.concat(dfs)

    # determine condition and update multiindex
    cond_idx = [((df["gfp_int"] > threshold).groupby("cell").mean() > 0.5).astype(int)]
    df["cond"] = np.broadcast_to(np.atleast_2d(np.array(conditions)).T, (2, 13))[
        tuple(cond_idx)
    ].ravel()
    # df["cond"] = np.array([[condition] * 13, ["N"] * 13])[tuple(cond_idx)].ravel()
    df.set_index("cond", append=True, inplace=True)
    df = df.reorder_levels(
        ["cond", "cell", "sub-cell"],
    )
    df["well"] = well_number
    return df, images
