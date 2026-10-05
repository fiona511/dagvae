"""Support functions for the baseline analyses (baselines.ipynb).

Sections
    1. Preprocessing      coordinates, seasonal means, area means, standardising
    2. EOF / clustering   small helpers kept from earlier notebooks
    3. Regression         cross-validated grid-point regression on indices
    4. MCA                maximum covariance analysis and its fold stability
    5. Composites         permutation significance of cluster composites
    6. Plotting           shared map set-up and all figure functions

Conventions: fields are xarray DataArrays with dims (season_year, lat, lon) after
seasonal averaging; longitudes are 0-360 (see to_360).

Earlier versions of this file defined djf_mean and on_mean twice (the second definition
silently replaced the first); only one version of each is kept here, see djf_mean.
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from sklearn.linear_model import LinearRegression


# ═════════════════════════════════════════════════════════════════════════════
# 1. Preprocessing
# ═════════════════════════════════════════════════════════════════════════════
def lat_band(da, south, north):
    """Latitude slice that works whether lat is ascending or descending."""
    asc = bool(da['lat'][0] < da['lat'][-1])
    return da.sel(lat=slice(south, north) if asc else slice(north, south))


def to_360(da):
    """Put longitudes on 0-360 (sorted) so Pacific boxes don't wrap."""
    if float(da['lon'].min()) < 0:
        da = da.assign_coords(lon=(da['lon'] % 360)).sortby('lon')
    return da


def prep(da):
    """0-360 longitudes, ascending lat and lon."""
    if float(da['lon'].min()) < 0:
        da = da.assign_coords(lon=(da['lon'] % 360))
    return da.sortby('lat').sortby('lon')


def season_year(da):
    """Label each month by the year of the December that opens its DJF season."""
    yr, mon = da['time'].dt.year, da['time'].dt.month
    return xr.where(mon == 12, yr, yr - 1).rename('season_year')


def djf_mean(da):
    """DJF mean labelled by the year of the December (DJF 1850/51 -> 1850).

    Seasons with fewer than 3 months (the truncated first/last season of the record) are
    dropped. (The second, overriding definition in the old file dropped the first and
    last season unconditionally, which removes a complete season if the record starts in
    December or ends in February. This count-based version is the one baselines.ipynb
    used.)
    """
    da = da.assign_coords(season_year=season_year(da))
    sel = da.sel(time=da['time'].dt.month.isin([12, 1, 2]))
    n = sel['time'].groupby(sel['season_year']).count()
    out = sel.groupby('season_year').mean('time')
    return out.where(n == 3, drop=True)


def on_mean(da):
    """October-November mean, labelled by calendar year (so it precedes DJF of that year)."""
    sel = da.sel(time=da['time'].dt.month.isin([10, 11]))
    return sel.groupby('time.year').mean('time').rename(year='season_year')


def wmean(da, dims=('lat', 'lon')):
    """cos(lat)-weighted mean over `dims` (NaNs, e.g. land, are skipped)."""
    return da.weighted(np.cos(np.deg2rad(da['lat']))).mean(dim=list(dims))


def area_mean(da):
    """cos(lat)-weighted mean over lat and lon."""
    return wmean(da, ('lat', 'lon'))


def standardize(da, dim='season_year'):
    """Per grid point: remove the mean along `dim` and divide by the std along `dim`."""
    return (da - da.mean(dim)) / da.std(dim)


def standardize_onestd(da, dim='season_year'):
    """Remove the mean along `dim`, then divide by ONE std taken over all dims (time and
    space), so regions of high variability keep their larger amplitude."""
    return (da - da.mean(dim)) / da.std()


def standardize_train_test(da, ref, dim='season_year'):
    """Standardise `da` with the per-grid-point mean and std of `ref` (training data)."""
    return (da - ref.mean(dim)) / ref.std(dim)


def calculate_anomalies(x, dim='time'):
    """Subtract the mean along `dim`."""
    return x - x.mean(dim=dim)


# ═════════════════════════════════════════════════════════════════════════════
# 2. EOF / clustering helpers
# ═════════════════════════════════════════════════════════════════════════════
def reshape_data_for_clustering(xarray_data):
    """[time, lat, lon] -> [time, lat*lon] numpy array.

    Uses Fortran order, i.e. latitude varies fastest along the flattened space axis;
    reshape back with order='F' as well.
    """
    data = xarray_data.values
    nt, ny, nx = data.shape
    return np.reshape(data, [nt, ny * nx], order='F')


def eof_analysis(dataset_xarray, pc_number, reconstruction_number):
    """Unweighted EOFs (time must be the first dim): EOFs, variance fractions, the first
    `pc_number` PCs and the field reconstructed from `reconstruction_number` EOFs."""
    from eofs.xarray import Eof          # imported here so eofs is only needed for this
    solver = Eof(dataset_xarray, center=True)
    return (solver.eofs(), solver.varianceFraction(), solver.pcs(npcs=pc_number),
            solver.reconstructedField(reconstruction_number))


# ═════════════════════════════════════════════════════════════════════════════
# 3. Cross-validated grid-point regression on indices
# ═════════════════════════════════════════════════════════════════════════════
def acc_map(obs, pred, dim='season_year'):
    """Grid-point anomaly correlation over `dim`."""
    o = obs - obs.mean(dim)
    p = pred - pred.mean(dim)
    return (o * p).sum(dim) / np.sqrt((o ** 2).sum(dim) * (p ** 2).sum(dim))


def to_xr(X, yrs, model, lat, lon):
    """model.predict(X) as a (season_year, lat, lon) DataArray."""
    return xr.DataArray(model.predict(X).reshape(len(yrs), lat.size, lon.size),
                        coords={'season_year': yrs, 'lat': lat, 'lon': lon},
                        dims=['season_year', 'lat', 'lon'])


def cv_regression(predictors, target, years, n_folds=10, buffer=3, verbose=True):
    """Blocked cross-validation of a linear regression of `target` on `predictors`.

    The seasons in `years` are split into `n_folds` contiguous blocks. In each fold the
    model is TRAINED on one block and TESTED on all other seasons, except `buffer`
    seasons either side of the training block (positions in `years`, so if seasons were
    removed beforehand the gap can span more calendar years). Predictors and target are
    standardised per grid point with training-block statistics; no intercept is fitted
    (training data have zero mean after standardising).

    predictors  list of 1-D DataArrays over season_year (raw indices)
    target      (season_year, lat, lon) DataArray (raw field)
    Returns dict with coefs [folds, n_predictors, nlat, nlon], acc_train / acc_test
    (area-weighted mean ACC per fold), acc_maps (test ACC map per fold) and mean_acc_map.
    """
    target = target.transpose('season_year', 'lat', 'lon')
    lat, lon = target['lat'], target['lon']
    n = len(years)
    coef_folds, acc_train, acc_test, acc_map_folds = [], [], [], []

    for fold, tr_idx in enumerate(np.array_split(np.arange(n), n_folds)):
        lo, hi = tr_idx[0], tr_idx[-1]
        te_idx = np.setdiff1d(np.arange(n),
                              np.arange(max(0, lo - buffer), min(n, hi + buffer + 1)))
        tr, te = years[tr_idx], years[te_idx]

        Xtr = np.column_stack([standardize_train_test(v.sel(season_year=tr),
                                                      v.sel(season_year=tr)).values
                               for v in predictors])
        Xte = np.column_stack([standardize_train_test(v.sel(season_year=te),
                                                      v.sel(season_year=tr)).values
                               for v in predictors])

        t_ref = target.sel(season_year=tr)
        Ytr = standardize_train_test(t_ref, t_ref)
        Yte = standardize_train_test(target.sel(season_year=te), t_ref)

        model = LinearRegression(fit_intercept=False)
        model.fit(Xtr, Ytr.values.reshape(len(tr), -1))

        a_tr = acc_map(Ytr, to_xr(Xtr, tr, model, lat, lon))
        a_te = acc_map(Yte, to_xr(Xte, te, model, lat, lon))

        acc_train.append(float(area_mean(a_tr)))
        acc_test.append(float(area_mean(a_te)))
        acc_map_folds.append(a_te)
        coef_folds.append(model.coef_.T.reshape(len(predictors), lat.size, lon.size))

        if verbose:
            print(f'Fold {fold+1:02d}/{n_folds} — train {len(tr):4d} yr, test {len(te):4d} yr '
                  f'| ACC train {acc_train[-1]:+.3f} | test {acc_test[-1]:+.3f}')

    if verbose:
        print(f'\nACC (area-weighted, mean over folds): '
              f'train {np.mean(acc_train):+.3f} | test {np.mean(acc_test):+.3f} '
              f'± {np.std(acc_test):.3f}')
    return dict(coefs=np.array(coef_folds), acc_train=acc_train, acc_test=acc_test,
                acc_maps=acc_map_folds,
                mean_acc_map=xr.concat(acc_map_folds, dim='fold').mean('fold'))


def sign_agreement(x, ref=None):
    """Fraction of the first axis of `x` (folds) that shares a sign.

    ref=None: agreement with the majority sign, max(frac > 0, 1 - frac > 0).
    ref given: agreement with sign(ref) (e.g. the mean over folds).
    Note: in the majority version NaN and 0 count as negative.
    """
    if ref is None:
        pos = (x > 0).mean(0)
        return np.maximum(pos, 1 - pos)
    return (np.sign(x) == np.sign(ref)[None]).mean(0)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Maximum covariance analysis
# ═════════════════════════════════════════════════════════════════════════════
def mca(field1, field2, n_modes=3, dim='season_year'):
    """Maximum Covariance Analysis (SVD of the cross-covariance matrix) between two
    (season_year, lat, lon) DataArrays that share the same time axis.

    Both fields are area-weighted by sqrt(cos lat) before the SVD; grid points that are
    NaN at any time (e.g. land) are left out. The returned spatial patterns are
    homogeneous regression maps: each field's (unweighted) anomalies regressed onto its
    own standardised expansion coefficient, i.e. physical units per one std of the
    coefficient. Signs are fixed so that paired coefficients correlate positively.

    Returns dict: scf (squared covariance fraction per mode), corr (r between paired
    coefficients), s (singular values), A, B (standardised coefficients [time, mode]),
    reg1, reg2 (regression maps, dims mode, lat, lon).
    """
    # align on common seasons only — not lat/lon, which differ between fields
    field1, field2 = xr.align(field1, field2, join='inner', exclude=['lat', 'lon'])
    f1 = field1 - field1.mean(dim)
    f2 = field2 - field2.mean(dim)

    w1 = np.sqrt(np.cos(np.deg2rad(f1.lat)).clip(0))
    w2 = np.sqrt(np.cos(np.deg2rad(f2.lat)).clip(0))

    Xv = (f1 * w1).stack(space=('lat', 'lon')).transpose(dim, 'space').values
    Yv = (f2 * w2).stack(space=('lat', 'lon')).transpose(dim, 'space').values
    Xc = Xv[:, ~np.isnan(Xv).any(axis=0)]
    Yc = Yv[:, ~np.isnan(Yv).any(axis=0)]
    T = Xc.shape[0]

    C = (Xc.T @ Yc) / (T - 1)
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    V = Vt.T
    scf = s**2 / np.sum(s**2)

    A = Xc @ U
    B = Yc @ V
    A_std = A / A.std(axis=0, ddof=1)
    B_std = B / B.std(axis=0, ddof=1)

    corr = np.array([np.corrcoef(A_std[:, k], B_std[:, k])[0, 1] for k in range(len(s))])

    # SVD signs are arbitrary per mode; fix so paired coefficients correlate positively
    flip = np.where(corr < 0, -1.0, 1.0)
    B_std, corr = B_std * flip, corr * flip

    def regmap(anom, coeff_std, n):
        stacked = anom.stack(space=('lat', 'lon')).transpose(dim, 'space')
        maps = coeff_std[:, :n].T @ stacked.values / (stacked.shape[0] - 1)
        return xr.DataArray(maps, dims=('mode', 'space'),
                            coords={'mode': np.arange(1, n + 1),
                                    'space': stacked.space}).unstack('space')

    n_modes = min(n_modes, len(s))
    return {'scf': scf, 'corr': corr, 's': s, 'A': A_std, 'B': B_std,
            'reg1': regmap(f1, A_std, n_modes), 'reg2': regmap(f2, B_std, n_modes)}


def mca_folds(f1, f2, n_folds=3, n_modes=3, dim='season_year'):
    """MCA separately on `n_folds` contiguous blocks of the record."""
    f1, f2 = xr.align(f1, f2, join='inner', exclude=['lat', 'lon'])
    n = f1.sizes[dim]
    return [mca(f1.isel({dim: idx}), f2.isel({dim: idx}), n_modes=n_modes, dim=dim)
            for idx in np.array_split(np.arange(n), n_folds)]


def match_to_full(fold, full, n_modes=3):
    """Flip each fold's mode k to the sign convention of full-record mode k (judged on
    the field-1 pattern). Returns stacked (reg1, reg2) maps [mode, lat, lon].

    Limitation: modes are matched by index only. If two modes swap order between a fold
    and the full record (likely when their SCFs are small and similar), mode k of the
    fold is compared with a different pattern.
    """
    r1 = np.stack([fold['reg1'].isel(mode=k).values for k in range(n_modes)])
    r2 = np.stack([fold['reg2'].isel(mode=k).values for k in range(n_modes)])
    for k in range(n_modes):
        a, b = full['reg1'].isel(mode=k).values, r1[k]
        ok = np.isfinite(a) & np.isfinite(b)
        if np.sum(a[ok] * b[ok]) < 0:
            r1[k], r2[k] = -r1[k], -r2[k]
    return r1, r2


# ═════════════════════════════════════════════════════════════════════════════
# 5. Composites
# ═════════════════════════════════════════════════════════════════════════════
def composite_sig(anom, labels, n_clusters, n_perm=500, seed=0, alpha=0.05):
    """Grid-point significance of cluster composites by permuting the cluster labels.

    anom: (season_year, lat, lon) anomalies; labels: integer cluster label per season.
    A composite is significant where |random composite| >= |observed composite| in at
    most a fraction `alpha` of the permutations (two-sided; assumes zero-mean anomalies).
    No correction for testing many grid points. NaN cells come out True; mask them when
    plotting.
    """
    rng = np.random.default_rng(seed)
    obs = anom.groupby(labels.rename('k')).mean('season_year')
    vals = anom.transpose('season_year', ...).values
    hits = np.zeros(obs.shape)
    for _ in range(n_perm):
        perm = rng.permutation(labels.values)
        null = np.stack([vals[perm == k].mean(0) for k in range(n_clusters)])
        hits += (np.abs(null) >= np.abs(obs.values))
    return xr.DataArray(hits / n_perm <= alpha, dims=obs.dims, coords=obs.coords)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Plotting
# ═════════════════════════════════════════════════════════════════════════════
DATA_CRS = ccrs.PlateCarree()


def _circular_boundary(ax):
    """Clip a polar-stereographic axis to a circle."""
    theta = np.linspace(0, 2 * np.pi, 200)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    ax.set_boundary(mpath.Path(verts * 0.5 + 0.5), transform=ax.transAxes)


def cyclic(vals, lon):
    """Close the longitude seam, but only for (near-)global grids. For a regional grid,
    add_cyclic_point would append a copy of the western edge at the eastern edge."""
    if np.ptp(lon) > 340:
        return add_cyclic_point(vals, coord=lon, axis=-1)
    return vals, lon


def setup_map_ax(ax, field, proj, polar_lat=-30, gridlines=True):
    """Coastlines, gridlines and extent: circular cap down to `polar_lat` for
    SouthPolarStereo, otherwise the field's lat/lon range."""
    ax.coastlines(linewidth=0.5)
    if gridlines:
        ax.gridlines(linewidth=0.3, color='gray', alpha=0.5)
    if isinstance(proj, ccrs.SouthPolarStereo):
        ax.set_extent([-180, 180, -90, polar_lat], crs=DATA_CRS)
        _circular_boundary(ax)
    else:
        ax.set_extent([float(field['lon'].min()), float(field['lon'].max()),
                       float(field['lat'].min()), float(field['lat'].max())], crs=DATA_CRS)


def plot_coef_maps(coefs, stipple, lat, lon, names, suptitle, polar_lat=-30):
    """South-polar maps of regression coefficients (one per predictor) with stippling.

    coefs, stipple: [n_predictors, nlat, nlon]; one shared, symmetric colour scale.
    """
    fig, axes = plt.subplots(1, len(names), figsize=(16, 6),
                             subplot_kw={'projection': ccrs.SouthPolarStereo()})
    vmax = float(np.abs(coefs).max())
    levels = np.linspace(-vmax, vmax, 21)
    lon2d, lat2d = np.meshgrid(lon.values, lat.values)

    for i, name in enumerate(names):
        ax = axes[i]
        ax.set_extent([-180, 180, -90, polar_lat], crs=DATA_CRS)
        _circular_boundary(ax)
        d, lonc = cyclic(coefs[i], lon.values)
        im = ax.contourf(lonc, lat.values, d, levels=levels, cmap='RdBu_r',
                         extend='both', transform=DATA_CRS)
        m = stipple[i]
        ax.scatter(lon2d[m], lat2d[m], s=0.8, c='k', alpha=0.5, transform=DATA_CRS, zorder=5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.set_title(name, fontsize=12)
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.85)

    fig.suptitle(suptitle, fontsize=12, y=1.04)
    plt.tight_layout()
    plt.show()
    return fig


def plot_mca(res, name1='ua850 (SH jet)', name2='ts (tropical Pacific)', n=3,
             stip1=None, stip2=None, suptitle=None):
    """Homogeneous regression maps of the first n MCA modes (field 1 polar, field 2
    tropical), one symmetric colour scale per field (98th percentile of |map|).
    With stip1 / stip2 ([mode, lat, lon] bool) the maps are stippled instead of contoured."""
    stippled = stip1 is not None
    fig = plt.figure(figsize=(13, 3.4 * n), dpi=150)
    fields = [(res['reg1'], name1, ccrs.SouthPolarStereo(), stip1),
              (res['reg2'], name2, ccrs.PlateCarree(central_longitude=180), stip2)]
    scales = [np.linspace(-v, v, 13) for v in
              [np.nanpercentile(np.abs(f[0].isel(mode=slice(0, n)).values), 98) for f in fields]]

    for k in range(n):
        for j, (reg, name, proj, stip) in enumerate(fields):
            field = reg.isel(mode=k)
            lat, lon, vals = field.lat.values, field.lon.values, field.values
            ax = fig.add_subplot(n, 2, k * 2 + j + 1, projection=proj)

            pv, plon = cyclic(vals, lon)
            cf = ax.contourf(plon, lat, pv, levels=scales[j], cmap='RdBu_r',
                             extend='both', transform=DATA_CRS)
            if stippled:
                lon2d, lat2d = np.meshgrid(lon, lat)
                m = stip[k] & np.isfinite(vals)
                ax.scatter(lon2d[m], lat2d[m], s=0.6, c='k', alpha=0.5,
                           transform=DATA_CRS, zorder=5)
            else:
                ax.contour(plon, lat, pv, levels=scales[j], colors='k', linewidths=0.35,
                           transform=DATA_CRS)

            setup_map_ax(ax, field, proj)
            fig.colorbar(cf, ax=ax, shrink=0.65, pad=0.05, aspect=20, orientation='vertical')
            ax.set_title(f'{name}  mode {k+1}\nSCF = {res["scf"][k]*100:.1f}%   '
                         f'r = {res["corr"][k]:+.2f}', fontsize=10)

    if suptitle:
        fig.suptitle(suptitle, y=1.01, fontsize=11)
    plt.tight_layout()
    plt.show()
    return fig


def plot_mca_stippled(full, stip1, stip2, n=3, n_folds=None,
                      name1='ua850 (SH jet)', name2='ts (tropical Pacific)'):
    """plot_mca with stippling where the sign agrees in all folds.
    (Previously read the global N_FOLDS for its title; now passed in.)"""
    title = 'MCA modes (full record), stippling: sign agrees in all ' \
            + (f'{n_folds} folds' if n_folds else 'folds')
    return plot_mca(full, name1, name2, n, stip1=stip1, stip2=stip2, suptitle=title)


def plot_coefficient_correlations(res, n=5, labels=('ua850', 'ts')):
    """Correlation matrices of the MCA expansion coefficients (within and across fields)."""
    A, B = res['A'][:, :n], res['B'][:, :n]
    C = np.corrcoef(np.column_stack([A, B]).T)
    blocks = [(C[:n, :n], f'{labels[0]} vs {labels[0]}'),
              (C[n:, n:], f'{labels[1]} vs {labels[1]}'),
              (C[:n, n:], f'{labels[0]} vs {labels[1]}')]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    ticks, names = np.arange(n), [f'{k+1}' for k in range(n)]
    for ax, (M, title) in zip(axes, blocks):
        im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(ticks, names)
        ax.set_yticks(ticks, names)
        ax.set_xlabel('mode')
        ax.set_ylabel('mode')
        ax.set_title(title, fontsize=10)
        for i in range(n):
            for j in range(n):
                v = M[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(v) > 0.5 else 'black')
        fig.colorbar(im, ax=ax, shrink=0.8, label='r')
    plt.tight_layout()
    plt.show()
    return C


def plot_cluster_composites(comp_ts, comp_ua, sig_ts, sig_ua, counts, n_total, n_clusters):
    """Two rows of cluster composites (SST tropical, ua850 polar), stippled where
    significant, one colour bar per row (98th percentile of |composite|)."""
    fig = plt.figure(figsize=(3.4 * n_clusters, 8.5), dpi=150)
    v1 = np.nanpercentile(np.abs(comp_ts.values), 98)
    v2 = np.nanpercentile(np.abs(comp_ua.values), 98)
    rows = [(comp_ts, sig_ts, ccrs.PlateCarree(central_longitude=180),
             np.linspace(-v1, v1, 13), 'SST [K]'),
            (comp_ua, sig_ua, ccrs.SouthPolarStereo(), np.linspace(-v2, v2, 13), 'ua850 [m/s]')]

    for i, (comp, sig, proj, levels, unit) in enumerate(rows):
        cf = None
        for k in range(n_clusters):
            f = comp.isel(k=k)
            ax = fig.add_subplot(2, n_clusters, i * n_clusters + k + 1, projection=proj)
            vals, plon = cyclic(f.values, f['lon'].values)
            cf = ax.contourf(plon, f['lat'].values, vals, levels=levels,
                             cmap='RdBu_r', extend='both', transform=DATA_CRS)
            lon2d, lat2d = np.meshgrid(f['lon'].values, f['lat'].values)
            m = sig.isel(k=k).values & np.isfinite(f.values)
            ax.scatter(lon2d[m], lat2d[m], s=0.4, c='k', alpha=0.4, transform=DATA_CRS, zorder=5)
            setup_map_ax(ax, f, proj)
            if i == 0:
                ax.set_title(f'Cluster {k+1}\nn = {counts[k]} ({counts[k]/n_total*100:.0f}%)',
                             fontsize=10)
        # one shared colorbar per row, to the right
        cax = fig.add_axes([0.92, 0.55 - i * 0.44, 0.012, 0.32])
        fig.colorbar(cf, cax=cax, label=unit)

    fig.text(0.085, 0.72, 'SST', rotation=90, va='center', fontsize=11)
    fig.text(0.085, 0.28, 'ua850', rotation=90, va='center', fontsize=11)
    fig.suptitle(f'k-means on DJF tropical Pacific SST (k={n_clusters}) with concurrent '
                 f'ua850 composites', y=0.97, fontsize=12)
    plt.subplots_adjust(left=0.11, right=0.90, top=0.88, bottom=0.04, wspace=0.08, hspace=0.05)
    plt.show()
    return fig


def plot_composite_pair(panels, suptitle, fill_levels=None):
    """Side-by-side composite maps, e.g. [(sst_comp, proj, unit), (ua_comp, proj, unit)].

    Contour lines use ±(98th percentile of |field|) per panel. Filled contours use the
    same levels, unless `fill_levels` is given (applied to every panel).
    """
    fig = plt.figure(figsize=(12, 4.5), dpi=150)
    for j, (f, proj, unit) in enumerate(panels):
        v = np.nanpercentile(np.abs(f.values), 98)
        line_levels = np.linspace(-v, v, 13)
        ax = fig.add_subplot(1, len(panels), j + 1, projection=proj)
        vals, plon = cyclic(f.values, f['lon'].values)
        cf = ax.contourf(plon, f['lat'].values, vals,
                         levels=line_levels if fill_levels is None else fill_levels,
                         cmap='RdBu_r', extend='both', transform=DATA_CRS)
        ax.contour(plon, f['lat'].values, vals, levels=line_levels,
                   colors='k', linewidths=0.3, transform=DATA_CRS)
        setup_map_ax(ax, f, proj)
        fig.colorbar(cf, ax=ax, shrink=0.7, pad=0.05, aspect=20, label=unit)
    fig.suptitle(suptitle, y=1.0, fontsize=12)
    plt.tight_layout()
    plt.show()
    return fig


def _map_row(n_panels, projection, figsize):
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, subplot_kw=dict(projection=projection),
                             squeeze=False)
    return fig, axes[0]


def visualise_contourplot_labels(cluster_centers, regime_names, vmin, vmax, steps, color_scheme,
                                 labels, col_number=5, borders=True,
                                 projection=ccrs.PlateCarree(central_longitude=180)):
    """One map per cluster centre (cluster, lat, lon), titled with regime name and frequency.

    Fixed: levels now include vmax and values outside [vmin, vmax] are coloured with the
    end colours (extend='both'); before, they were left blank.
    """
    nt = cluster_centers.shape[0]
    x, y = np.meshgrid(cluster_centers.lon, cluster_centers.lat)
    fig, axes = _map_row(col_number, projection, (14, 5))
    for i in range(nt):
        ax = axes[i]
        ax.contourf(x, y, cluster_centers[i, :, :], levels=np.arange(vmin, vmax + steps, steps),
                    transform=DATA_CRS, cmap=color_scheme, extend='both')
        ax.coastlines()
        if borders:
            ax.add_feature(cartopy.feature.BORDERS)
        ax.set_title('{}, {:4.1f}%'.format(regime_names[i], 100 * np.mean(np.asarray(labels) == i)))
    plt.tight_layout()
    return fig


def visualise_contourplot_latent_legend(samples, vmin, vmax, steps, color_scheme,
                                        col_number=5, borders=True,
                                        projection=ccrs.PlateCarree(central_longitude=180),
                                        unit_name='[mm/day]', shrink_value=0.4):
    """One map per sample (sample, lat, lon) in a row, with a shared colour bar."""
    nt = samples.shape[0]
    x, y = np.meshgrid(samples.lon, samples.lat)
    fig, axes = _map_row(col_number, projection, (1.5 + 3 * col_number, 5))
    cs = None
    for i in range(nt):
        ax = axes[i]
        cs = ax.contourf(x, y, samples[i, :, :], levels=np.arange(vmin, vmax + steps, steps),
                         transform=DATA_CRS, cmap=color_scheme)
        ax.coastlines()
        if borders:
            ax.add_feature(cartopy.feature.BORDERS)
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    cbar = fig.colorbar(cs, ax=list(axes), orientation='vertical', shrink=shrink_value, pad=0.02)
    cbar.set_label(unit_name)
    return fig
