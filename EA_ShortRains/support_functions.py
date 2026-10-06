#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
support_functions.py
====================

Helper functions:

    1. Anomalies & reshaping      - anomaly calculations, (time, lat, lon) <-> (time, space)
    2. Climate indices            - Nino3.4 and Dipole Mode Index (DMI) from SST files
    3. Train / test splitting     - consecutive block split + flattening
    4. Clustering                 - k-means regimes and centroid reshaping
    5. EOF analysis               - wrapper around the `eofs` package
    6. Model evaluation           - Keras history summary, R^2
    7. Latent-space reconstruction- decode latent samples back to (time, lat, lon)
    8. Plotting                   - contour-map panels (with/without titles, significance)
    9. Colour normalisation       - MidpointNormalize

Flattening convention
---------------------
Every spatial field is flattened with Fortran order (order='F'), i.e.
flat index = lat_index + n_lat * lon_index. All reshape functions below
use the same convention, so fields can be round-tripped safely.

@author: fionaspuler
"""

# =============================================================================
# Imports
# =============================================================================

import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import cartopy
import cartopy.crs as ccrs

from sklearn.cluster import KMeans
from eofs.xarray import Eof


# =============================================================================
# 1. Anomalies & reshaping
# =============================================================================

def calculate_anomalies(x):
    """Anomalies relative to the mean over the 'time' dimension."""
    return x - x.mean(dim='time')


def calculate_anomalies_year(x):
    """Anomalies relative to the mean over the 'year' dimension."""
    return x - x.mean(dim='year')


def calculate_anomalies_runningwindow(x, rolling_window=30):
    """Anomalies relative to a centred running mean of `rolling_window` time steps."""
    return x - x.rolling(time=rolling_window, min_periods=1, center=True).mean()


def reshape_data_for_clustering(xarray_data):
    """
    Flatten a (time, lat, lon) DataArray to a (time, lat*lon) numpy array
    using Fortran order (see module docstring).
    """
    data = xarray_data.values
    nt, ny, nx = data.shape
    return np.reshape(data, [nt, ny * nx], order='F')


# =============================================================================
# 2. Climate indices (Nino3.4, DMI)
# =============================================================================

def _load_sst(filename_sstdata, years=range(1940, 2024), trim_last=15):
    """
    Load the 'sst' variable, convert longitudes to 0-360, sort both
    coordinates ascending, keep the requested years and drop the last
    `trim_last` time steps.
    """
    sst = xr.open_dataset(filename_sstdata)['sst']
    sst.coords['longitude'] = np.mod(sst['longitude'], 360)
    sst = sst.sortby('longitude').sortby('latitude')
    sst = sst.sel(time=np.isin(sst.time.dt.year, list(years)))
    if trim_last:
        # positional trim on the time dimension (original assumed time was dim 0)
        sst = sst.isel(time=slice(None, -trim_last))
    return sst


def _detrend_against_gmst(index, gmst_data):
    """Remove the linear regression of `index` on global-mean surface temperature."""
    gmst = np.asarray(gmst_data)
    if gmst.shape[0] != index.shape[0]:
        raise ValueError(
            f"gmst_data has {gmst.shape[0]} values but the index has "
            f"{index.shape[0]} years - they must cover the same years."
        )
    coefs = poly.polyfit(gmst, index.values, 1)
    fit = poly.polyval(gmst, coefs)
    return index - fit


def calculate_nino34(filename_sstdata, gmst_data, detrend=True,
                     selected_months=[10, 11, 12], smooth_window_days=150,
                     trim_last=15):
    """
    Nino3.4 index (5S-5N, 170W-120W), averaged over `selected_months`
    for each year and standardised. Optionally detrended against GMST.

    Returns a DataArray with dimension 'year'.
    """
    sst = _load_sst(filename_sstdata, trim_last=trim_last)

    # area mean over the Nino3.4 box
    nino34 = sst.sel(longitude=slice(190, 240), latitude=slice(-5, 5))
    nino34 = nino34.mean(dim=['latitude', 'longitude'])

    if smooth_window_days:
        nino34 = nino34.rolling(time=smooth_window_days, min_periods=1, center=True).mean()

    # remove the seasonal cycle (DJF/MAM/JJA/SON means)
    nino34 = nino34.groupby('time.season').map(calculate_anomalies)

    if detrend:
        nino34 = _detrend_against_gmst(nino34, gmst_data)

    # seasonal average for each year, then standardise
    nino34 = nino34.sel(time=np.isin(nino34.time.dt.month, selected_months))
    nino34 = nino34.groupby('time.year').mean()
    nino34 = nino34 / nino34.std()

    return nino34


def calculate_dmi(filename_sstdata, gmst_data, detrend=True,
                  selected_months=[10, 11, 12], trim_last=15):
    """
    Dipole Mode Index: west box (10S-10N, 50E-70E) minus east box
    (10S-0, 90E-110E), averaged over `selected_months` per year and
    standardised. Optionally detrended against GMST.

    Note: no climatology is removed, so with detrend=False the index is
    scaled but NOT centred on zero (the detrending intercept centres it).
    """
    sst = _load_sst(filename_sstdata, trim_last=trim_last)

    west = sst.sel(longitude=slice(50, 70), latitude=slice(-10, 10))
    west = west.mean(dim=['latitude', 'longitude'])

    east = sst.sel(longitude=slice(90, 110), latitude=slice(-10, 0))
    east = east.mean(dim=['latitude', 'longitude'])

    dmi = west - east

    if detrend:
        dmi = _detrend_against_gmst(dmi, gmst_data)

    dmi = dmi.sel(time=np.isin(dmi.time.dt.month, selected_months))
    dmi = dmi.groupby('time.year').mean()
    dmi = dmi / dmi.std()

    return dmi


# =============================================================================
# 3. Train / test splitting
# =============================================================================

def split_and_reshape(da, train_idx, test_idx):
    """
    Split a (time, lat, lon) DataArray into train/test blocks and flatten each.

    Returns
    -------
    train_2d_xr, test_2d_xr : xr.DataArray, (time, lat, lon)
    train_flat, test_flat   : np.ndarray, (time, lat*lon), Fortran order
    """
    train_2d_xr = da.isel(time=train_idx)
    test_2d_xr = da.isel(time=test_idx)

    train_flat = reshape_data_for_clustering(train_2d_xr)
    test_flat = reshape_data_for_clustering(test_2d_xr)

    return train_2d_xr, test_2d_xr, train_flat, test_flat


# =============================================================================
# 4. Clustering
# =============================================================================

def calculate_clusters(xarray_data, cluster_number, calculation_steps=50):
    """
    Fit k-means on a (time, lat, lon) DataArray.
    `calculation_steps` is passed to KMeans as n_init (number of restarts).
    """
    data = reshape_data_for_clustering(xarray_data)
    return KMeans(n_clusters=cluster_number, n_init=calculation_steps,
                  random_state=0).fit(data)


def reshape_centroids_kmeans(kmeans, dataset_xarray):
    """
    Reshape k-means centroids back to (k, lat, lon), borrowing coordinates
    from the first k time steps of `dataset_xarray`.
    """
    k = kmeans.n_clusters
    nt, ny, nx = dataset_xarray.values.shape
    centroids = kmeans.cluster_centers_.reshape(k, ny, nx, order='F')

    template = dataset_xarray[0:k, :, :]
    return xr.DataArray(centroids, coords=template.coords,
                        dims=template.dims, attrs=template.attrs)


# =============================================================================
# 5. EOF analysis
# =============================================================================

def eof_analysis(dataset_xarray, pc_number):
    """
    EOF decomposition with the `eofs` package.

    Returns
    -------
    eofs, variance_fraction, pcs (first `pc_number`), reconstruction
    (field reconstructed from the first `pc_number` EOFs)
    """
    if Eof is None:
        raise ImportError("eof_analysis needs the 'eofs' package: pip install eofs")

    solver = Eof(dataset_xarray, center=True)

    eofs = solver.eofs()
    variance_fraction = solver.varianceFraction()
    pcs = solver.pcs(npcs=pc_number)
    reconstruction = solver.reconstructedField(pc_number)

    return eofs, variance_fraction, pcs, reconstruction


# =============================================================================
# 6. Model evaluation
# =============================================================================

# order in which the loss terms are reported (each followed by its 'val_' twin)
_HISTORY_KEYS = (
    ['loss']
    + [f'reconstruction_loss_z{j}' for j in (1, 2, 3)]
    + [f'regularization_loss_z{j}' for j in (1, 2, 3)]
)


def print_history(history_data):
    """
    Print the final-epoch training and validation losses of a Keras History
    object and return them as a one-row DataFrame.

    Column names keep the trailing space of the original version
    (e.g. 'loss ') so existing code that indexes them keeps working.
    """
    keys = [k for base in _HISTORY_KEYS for k in (base, 'val_' + base)]
    last = {k: history_data.history[k][-1] for k in keys}

    print(' '.join(f'{k} {v}' for k, v in last.items()))

    return pd.DataFrame({f'{k} ': [v] for k, v in last.items()})


def calculate_r2(y_true, y_pred):
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot."""
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - ss_res / ss_tot


# =============================================================================
# 7. Latent-space reconstruction
# =============================================================================

def reconstruct_to_xr(np_data, inputdim1, inputdim2, reference_data):
    """
    Turn flat (n, lat*lon) data (Fortran-order flattening) back into an
    (n, lat, lon) DataArray.

    inputdim1 = number of latitudes, inputdim2 = number of longitudes.
    Coordinates are taken from the first n time steps of `reference_data`,
    so it must have at least n time steps.
    """
    n = np_data.shape[0]
    # C-order reshape to (n, lon, lat) + transpose == Fortran-order unflatten
    reconstructed = np_data.reshape(n, inputdim2, inputdim1)
    reconstructed = np.transpose(reconstructed, (0, 2, 1))

    template = reference_data[0:n, :, :]
    return xr.DataArray(reconstructed, coords=template.coords,
                        dims=template.dims, attrs=template.attrs)


def reconstruct_samples(sample_number, latent_data, latent_data_other,
                        inputdim1, inputdim2, chosen_dimension,
                        reference_data, original_data, decoder,
                        order=[0, 2, 1], element_number=1):
    """
    Traverse one latent dimension and decode the result.

    `chosen_dimension` of `latent_data` is swept linearly from its min to its
    max over `sample_number` steps; all other dimensions of `latent_data` are
    held at their mean. The three latent inputs [other_0, other_1, sampled]
    are reordered with `order` before being passed to `decoder`, and output
    number `element_number` is un-flattened and de-standardised with the mean
    and std of `original_data`.

    Notes
    -----
    - The OTHER latent spaces are NOT held fixed: the first `sample_number`
      rows of each are used, so they change from sample to sample. If you
      want a clean traversal, replace them by their mean (see comment below).
    - `reference_data` needs at least `sample_number` time steps.
    """
    sampled_ls = []
    for i in range(latent_data.shape[1]):
        if i == chosen_dimension:
            sampled_ls.append(np.linspace(latent_data[:, i].min(),
                                          latent_data[:, i].max(),
                                          sample_number))
        else:
            sampled_ls.append(np.full(sample_number, latent_data[:, i].mean()))
    sampled = np.stack(sampled_ls, axis=1)          # (sample_number, latent_dim)

    # For a traversal with the other latent spaces fixed, use e.g.
    #   np.tile(latent_data_other[0].mean(axis=0), (sample_number, 1))
    latent_inputs = [latent_data_other[0][:sample_number, :],
                     latent_data_other[1][:sample_number, :],
                     sampled]
    latent_inputs = [latent_inputs[i] for i in order]

    z_list = decoder.predict(latent_inputs, verbose=0)
    z_sampled = z_list[element_number]

    reconstructed_xr = reconstruct_to_xr(np_data=z_sampled, inputdim1=inputdim1,
                                         inputdim2=inputdim2,
                                         reference_data=reference_data)

    # undo the standardisation used for training
    return reconstructed_xr * original_data.std() + original_data.mean()


# =============================================================================
# 8. Plotting
# =============================================================================

def _contour_panels(fields, vmin, vmax, steps, color_scheme, col_number,
                    borders, projection, figsize,
                    titles=None, significance=None):
    """
    Shared engine for the map plots: one filled-contour map per entry along
    the first dimension of `fields`, laid out in a single row.

    Returns (fig, axes, last_contour_set).
    """
    fields = fields.transpose(..., 'latitude', 'longitude')
    nt = fields.shape[0]
    if nt > col_number:
        raise ValueError(f"{nt} fields to plot but only col_number={col_number} panels.")

    x, y = np.meshgrid(fields.longitude, fields.latitude)
    levels = np.arange(vmin, vmax + steps, steps)

    # squeeze=False keeps `axes` a 2-D array even when col_number == 1
    fig, axes = plt.subplots(1, col_number, figsize=figsize, squeeze=False,
                             subplot_kw=dict(projection=projection))
    axes = axes.ravel()

    cs = None
    for i in range(nt):
        ax = axes[i]
        cs = ax.contourf(x, y, fields[i, :, :], levels=levels,
                         transform=ccrs.PlateCarree(), cmap=color_scheme)

        if significance is not None:
            mask = np.asarray(significance[i]) == 1
            # transform is required: the map may be centred on 180 degrees
            ax.plot(x[mask], y[mask], 'k.', markersize=1,
                    transform=ccrs.PlateCarree(), label='Significant')

        ax.coastlines()
        if borders:
            ax.add_feature(cartopy.feature.BORDERS)
        if titles is not None:
            ax.set_title(titles[i])

    return fig, axes, cs


def _add_colorbar(fig, axes, cs, unit_name, shrink_value):
    """Tight layout leaving room on the right, plus one shared vertical colorbar."""
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    cbar = fig.colorbar(cs, ax=list(axes), orientation='vertical',
                        shrink=shrink_value, pad=0.02)
    cbar.set_label(unit_name)


def visualise_contourplot_latent_legend(samples, vmin, vmax, steps, color_scheme,
                                        col_number=5, borders=True,
                                        projection=ccrs.PlateCarree(central_longitude=180),
                                        unit_name='[mm/day]', shrink_value=0.4):
    """Row of maps (e.g. latent traversal samples) with a shared colorbar."""
    fig, axes, cs = _contour_panels(samples, vmin, vmax, steps, color_scheme,
                                    col_number, borders, projection,
                                    figsize=(1.5 + 3 * col_number, 5))
    _add_colorbar(fig, axes, cs, unit_name, shrink_value)
    return fig


def visualise_contourplot_latent_legend_significance(samples, samples_significance,
                                                     vmin, vmax, steps, color_scheme,
                                                     col_number=5, borders=True,
                                                     projection=ccrs.PlateCarree(central_longitude=180),
                                                     unit_name='[mm/day]', shrink_value=0.4):
    """
    As visualise_contourplot_latent_legend, with black dots where
    `samples_significance` == 1 (same shape as `samples`).
    """
    fig, axes, cs = _contour_panels(samples, vmin, vmax, steps, color_scheme,
                                    col_number, borders, projection,
                                    figsize=(1.5 + 3 * col_number, 5.5),
                                    significance=samples_significance)
    _add_colorbar(fig, axes, cs, unit_name, shrink_value)
    return fig


def _regime_titles(regime_names, labels, n):
    """'<name>, <frequency>%' for each of the first n regimes."""
    labels = np.asarray(labels)
    return [f"{regime_names[i]}, {100 * np.sum(labels == i) / len(labels):.1f}%"
            for i in range(n)]


def visualise_contourplot_labels(cluster_centers, regime_names, vmin, vmax, steps,
                                 color_scheme, labels, col_number=5, borders=True,
                                 projection=ccrs.PlateCarree(central_longitude=180)):
    """Row of regime (cluster-centre) maps titled with their frequency; no colorbar."""
    titles = _regime_titles(regime_names, labels, cluster_centers.shape[0])
    fig, axes, cs = _contour_panels(cluster_centers, vmin, vmax, steps, color_scheme,
                                    col_number, borders, projection,
                                    figsize=(14, 5), titles=titles)
    plt.tight_layout()
    return fig


def visualise_contourplot_labels_legend(cluster_centers, regime_names, vmin, vmax,
                                        steps, color_scheme, labels, col_number=5,
                                        borders=True,
                                        projection=ccrs.PlateCarree(central_longitude=180),
                                        unit_name='[mm/day]', shrink_value=0.4):
    """Row of regime maps titled with their frequency, plus a shared colorbar."""
    titles = _regime_titles(regime_names, labels, cluster_centers.shape[0])
    fig, axes, cs = _contour_panels(cluster_centers, vmin, vmax, steps, color_scheme,
                                    col_number, borders, projection,
                                    figsize=(1.5 + 3 * col_number, 5), titles=titles)
    _add_colorbar(fig, axes, cs, unit_name, shrink_value)
    return fig


# =============================================================================
# 9. Colour normalisation
# =============================================================================

class MidpointNormalize(Normalize):
    """
    Colour normalisation with a custom centre: vmin -> 0, vcenter -> 0.5,
    vmax -> 1 (piecewise linear). Useful for diverging colormaps with
    asymmetric ranges. vmin, vmax and vcenter must all be given.
    (matplotlib.colors.TwoSlopeNorm is the built-in equivalent.)
    """

    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # masked values / clipping are ignored; extrapolates beyond vmin/vmax
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
        return np.ma.masked_array(np.interp(value, x, y,
                                            left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)


