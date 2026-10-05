"""Preprocessing for the DAG-VAE: raw monthly model output -> standardised network inputs.

The model sees three things per season (one sample = one season):

    x1  a gridded "driver" field          (default: DJF tos, tropical Pacific, ocean only)
    x2  a gridded "response" field        (default: DJF ua850, 65S-35S)
    u   one or more scalar covariates     (default: Oct-Nov SPV index, u50 averaged 65S-50S)

Everything that used to be hard-coded (files, variables, months, lat/lon box, grid
resolution, latitude weighting) is now set in a preprocessing config dict. Start from
`DEFAULT_PREP` and override what you need, e.g.

    import copy, preprocessing as pp
    prep = copy.deepcopy(pp.DEFAULT_PREP)
    prep['x1']['months'] = 'SON'                    # or [9, 10, 11]
    prep['x1']['lon'] = (120, 290)
    prep['x2']['resolution'] = (2.5, 2.5)           # interpolate onto a 2.5 x 2.5 deg grid
    data = pp.load_and_preprocess('data/cesm2/', prep)

Field spec keys (x1, x2 and each entry in `covariates`):

    file        file name, relative to `datapath`
    var         variable name in the file
    months      months to average over. Ints in chronological order ([12, 1, 2]) or a
                string of consecutive month initials ('DJF', 'ON', 'NDJFM'). Each season is
                labelled with the calendar year of its FIRST month, so DJF 1850/51 -> 1850
                and Oct-Nov 1850 -> 1850 (i.e. the ON covariate precedes the DJF season).
                Only seasons with every month present are kept.
    lat         (south, north) in degrees, or None for all latitudes
    lon         (west, east) in degrees, 0-360 or -180-180, or None for all longitudes.
                Boxes across the Greenwich meridian are fine, e.g. (300, 30) or (-60, 30).
    level       pressure level to select if the file has a 'plev' dimension with more
                than one level (nearest match, in the file's units). Optional.
    resolution  (dlat, dlon) in degrees to linearly interpolate onto a regular grid within
                the box, or None (default) to keep the native grid. Optional.
    weighting   x1/x2 only: None, 'coslat' or 'sqrt_coslat'. Applied AFTER standardising,
                so each grid cell's contribution to the MSE loss scales with its area
                ('sqrt_coslat' -> squared error scales with cos(lat)). Optional.
    name        covariates only: a label used in printouts. Optional.

Grid cells that are missing (NaN, e.g. land in an ocean field) or constant at any time in
the training period are dropped from the network input; `FieldGrid.to_grid` puts them back
as NaN for plotting.
"""
import copy
import os

import numpy as np
import xarray as xr


# ═════════════════════════════════════════════════════════════════════════════
# Default configuration (reproduces the September 2026 setup)
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_PREP = dict(
    test_frac=0.2,          # last 20 % of seasons (contiguous) = test set
    x1=dict(
        file='tos_Omon_CESM2_piControl_r1i1p1f1_gr.nc', var='tos',
        months=[12, 1, 2], lat=(-20, 20), lon=(160, 285),
        resolution=None, weighting=None,
    ),
    x2=dict(
        file='ua850_Amon_CESM2_piControl_r1i1p1f1_gn.nc', var='ua',
        months=[12, 1, 2], lat=(-65, -35), lon=None,
        resolution=None, weighting='sqrt_coslat',
    ),
    covariates=[
        dict(name='spv', file='ua50_Amon_CESM2_piControl_r1i1p1f1_gn.nc', var='ua',
             months=[10, 11], lat=(-65, -50), lon=None),
    ],
)

_MONTH_INITIALS = 'JFMAMJJASOND'


# ═════════════════════════════════════════════════════════════════════════════
# Small, general helpers
# ═════════════════════════════════════════════════════════════════════════════
def parse_months(months):
    """Turn a month specification into a list of ints (1-12) in chronological order.

    Accepts an int (3), a list of ints ([12, 1, 2]) or a string of consecutive month
    initials ('DJF', 'JJA', 'ON', 'NDJFM'). A string is matched against the calendar
    wrapped round once, so 'DJF' -> [12, 1, 2]. For a single month, pass an int.
    """
    if isinstance(months, str):
        s = months.upper()
        idx = (_MONTH_INITIALS * 2).find(s)
        if idx < 0 or not 1 <= len(s) <= 12:
            raise ValueError(f'{months!r} is not a sequence of consecutive month initials')
        return [(idx + i) % 12 + 1 for i in range(len(s))]
    months = [int(m) for m in np.atleast_1d(months)]
    if not months or any(m < 1 or m > 12 for m in months) or len(set(months)) != len(months):
        raise ValueError(f'months must be unique ints in 1..12, got {months}')
    # offsets from the first month must increase, i.e. the list is in chronological order
    offsets = [(m - months[0]) % 12 for m in months]
    if offsets != sorted(offsets):
        raise ValueError(f'list months in chronological order starting with the first month '
                         f'of the season, got {months}')
    return months


def standardise_coords(da):
    """Rename latitude/longitude -> lat/lon, put lon on 0-360 and sort lat and lon ascending."""
    da = da.rename({k: v for k, v in (('latitude', 'lat'), ('longitude', 'lon')) if k in da.dims})
    if float(da['lon'].min()) < 0:
        da = da.assign_coords(lon=(da['lon'] % 360))
    return da.sortby('lat').sortby('lon')


# kept under its old name so older notebooks keep working
prep = standardise_coords


def select_level(da, level=None):
    """Drop a 'plev' dimension: pick `level` (nearest) if given, else squeeze a single level."""
    if 'plev' not in da.dims:
        return da
    if level is not None:
        return da.sel(plev=level, method='nearest', drop=True) if da.sizes['plev'] > 1 \
            else da.squeeze('plev', drop=True)
    if da.sizes['plev'] != 1:
        raise ValueError("file has several pressure levels — set 'level' in the field spec")
    return da.squeeze('plev', drop=True)


def select_box(da, lat=None, lon=None):
    """Cut out a lat/lon box. `da` must have ascending lat and 0-360 lon (standardise_coords).

    lat: (south, north) or None.  lon: (west, east) or None; may cross 0E, e.g. (300, 30),
    in which case the longitudes east of 0E are relabelled 360-390 so the box stays
    contiguous and monotonic (cartopy handles lon > 360 fine).
    """
    if lat is not None:
        south, north = sorted(lat)
        da = da.sel(lat=slice(south, north))
    if lon is not None and (lon[1] - lon[0]) % 360 != 0:
        west, east = lon[0] % 360, lon[1] % 360
        if west <= east:
            da = da.sel(lon=slice(west, east))
        else:                                   # box crosses the Greenwich meridian
            keep = (da['lon'] >= west) | (da['lon'] <= east)
            da = da.isel(lon=np.flatnonzero(keep.values))
            da = da.assign_coords(lon=np.where(da['lon'] < west, da['lon'] + 360, da['lon']))
            da = da.sortby('lon')
    if da.sizes['lat'] == 0 or da.sizes['lon'] == 0:
        raise ValueError(f'empty selection for lat={lat}, lon={lon} — check the box')
    return da


def regrid(da, dlat, dlon, method='linear'):
    """Interpolate onto a regular dlat x dlon grid spanning the field's own lat/lon range.

    Linear interpolation propagates NaN: a target point next to a missing (e.g. land) cell
    becomes NaN and is then dropped from the network input.
    """
    lat = np.arange(float(da['lat'].min()), float(da['lat'].max()) + 1e-6, dlat)
    lon = np.arange(float(da['lon'].min()), float(da['lon'].max()) + 1e-6, dlon)
    return da.interp(lat=lat, lon=lon, method=method)


def season_labels(time, months):
    """Season year of each time step: the calendar year of the season's first month."""
    months = parse_months(months)
    year, mon = time.dt.year.values, time.dt.month.values
    # months numerically before the first listed month belong to the season that
    # started in the previous calendar year (e.g. Jan/Feb of a DJF season)
    return year - (mon < months[0]).astype(int)


def seasonal_mean(da, months, full_seasons_only=True):
    """Average over the given months, one value per season, along a new 'season_year' dim.

    months: see parse_months, e.g. [12, 1, 2] or 'DJF'. Labelled by the year of the first
    month (DJF 1850/51 -> 1850). With full_seasons_only, seasons that are cut off at the
    start or end of the record are dropped.
    """
    months = parse_months(months)
    in_season = np.isin(da['time'].dt.month.values, months)
    da = da.isel(time=in_season)
    labels = season_labels(da['time'], months)
    da = da.assign_coords(season_year=('time', labels))
    out = da.groupby('season_year').mean('time')
    if full_seasons_only:
        years, counts = np.unique(labels, return_counts=True)
        out = out.sel(season_year=years[counts == len(months)])
    return out


def djf_mean(da):
    """DJF mean labelled by the year of the December; full seasons only."""
    return seasonal_mean(da, [12, 1, 2])


def on_mean(da):
    """October-November mean."""
    return seasonal_mean(da, [10, 11])


def area_mean(da, dims=('lat', 'lon')):
    """cos(lat)-weighted mean over `dims` (NaNs skipped)."""
    return da.weighted(np.cos(np.deg2rad(da['lat']))).mean(dim=list(dims))


# old name
wmean = area_mean


def _weight_from_coslat(coslat, kind):
    if kind is None:
        return xr.ones_like(coslat, dtype=float)
    if kind == 'coslat':
        return coslat
    if kind == 'sqrt_coslat':
        return np.sqrt(coslat)
    raise ValueError(f"unknown weighting {kind!r}; use None, 'coslat' or 'sqrt_coslat'")


def lat_weight(lat, kind):
    """Latitude weight applied to standardised fields: None, 'coslat' or 'sqrt_coslat'."""
    return _weight_from_coslat(np.cos(np.deg2rad(lat)), kind)


def standardize(da, dim='season_year'):
    """Remove the mean and divide by the standard deviation along `dim`."""
    return (da - da.mean(dim)) / da.std(dim)


def calculate_anomalies(x, dim='time'):
    """Subtract the mean along `dim`."""
    return x - x.mean(dim=dim)


def reshape_data_for_clustering(xarray_data):
    """[time, lat, lon] -> [time, lat*lon] numpy array.

    Note: uses Fortran (column-major) order, unlike the network inputs built below, which
    are C-ordered. Kept unchanged for compatibility with older clustering code.
    """
    data = xarray_data.values
    nt, ny, nx = data.shape
    return np.reshape(data, [nt, ny * nx], order='F')


def eof_analysis(dataset_xarray, pc_number, reconstruction_number):
    """EOFs, variance fractions, the first `pc_number` PCs and a reconstruction from the
    first `reconstruction_number` EOFs (eofs package; time must be the first dim)."""
    from eofs.xarray import Eof          # imported here so eofs is only needed for this
    solver = Eof(dataset_xarray, center=True)
    return (solver.eofs(), solver.varianceFraction(), solver.pcs(npcs=pc_number),
            solver.reconstructedField(reconstruction_number))


# ═════════════════════════════════════════════════════════════════════════════
# Containers handed to training and plotting
# ═════════════════════════════════════════════════════════════════════════════
class FieldGrid:
    """Everything needed to go between a flat network array and a (lat, lon) map.

    Attributes
        name       'x1' or 'x2'
        lat, lon   grid coordinates (DataArrays)
        mask       bool [nlat*nlon], True where the cell is part of the network input
        weighting  latitude weighting applied after standardising (see lat_weight)
        coslat     cos(lat) DataArray, for area-weighted means over maps
        area_weights  cos(lat) for each network feature (flat), for area-weighted scores
        std        (lat, lon) training-period std of the raw seasonal field, used to turn
                   standardised values back into physical anomalies (to_physical)
        units      units attribute of the raw variable ('' if the file has none)
    """

    def __init__(self, name, lat, lon, mask, weighting, std=None, units=''):
        self.name, self.lat, self.lon = name, lat, lon
        self.mask = np.asarray(mask, bool).ravel()
        self.weighting = weighting
        self.std, self.units = std, units
        self.nlat, self.nlon = lat.size, lon.size
        self.coslat = np.cos(np.deg2rad(lat))
        self.area_weights = np.repeat(self.coslat.values, self.nlon)[self.mask]

    @property
    def n_features(self):
        return int(self.mask.sum())

    def to_grid(self, flat, years=None):
        """[n, n_features] -> DataArray (season_year, lat, lon), dropped cells as NaN."""
        flat = np.atleast_2d(flat)
        full = np.full((flat.shape[0], self.mask.size), np.nan)
        full[:, self.mask] = flat
        years = np.arange(flat.shape[0]) if years is None else years
        return xr.DataArray(full.reshape(-1, self.nlat, self.nlon),
                            coords={'season_year': years, 'lat': self.lat, 'lon': self.lon},
                            dims=['season_year', 'lat', 'lon'])

    def unweight(self, da):
        """Undo the latitude weighting for plotting (guarded near the poles)."""
        if self.weighting is None:
            return da
        return da / _weight_from_coslat(np.maximum(self.coslat, 1e-2), self.weighting)

    def to_physical(self, da):
        """Network-space map -> physical anomaly (undo weighting, multiply by train std).

        Since the training mean is removed before standardising, the result is an anomaly
        relative to the training-period climatology (or a difference, for differences).
        """
        return self.unweight(da) * self.std
    def __repr__(self):
        return (f'FieldGrid({self.name}: {self.nlat} lat x {self.nlon} lon, '
                f'{self.n_features} active cells, weighting={self.weighting})')


class DataBundle:
    """Output of load_and_preprocess. Attributes can also be read dict-style (data['x1_train']).

    Network arrays (float32, standardised with training-period statistics):
        x1_train, x1_test   [n_seasons, x1.n_features]
        x2_train, x2_test   [n_seasons, x2.n_features]
        u_train, u_test     [n_seasons, n_covariates]
        years_train, years_test   season_year labels
    Grids:          x1, x2 (FieldGrid)
    Physical fields (seasonal means, before standardising):
        x1_raw, x2_raw (season_year, lat, lon), u_raw (season_year, covariate)
    Standardised (and weighted) fields on the grid: x1_n, x2_n, u_n
    Other:          prep (the config used), covariate_names, covariate_units, n_train
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self):
        return self.__dict__.keys()

    @property
    def n_covariates(self):
        return self.u_train.shape[1]


# ═════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════════════
def load_field(datapath, spec):
    """Open one variable and apply level / coordinate / box / regrid steps from its spec."""
    da = xr.open_dataset(os.path.join(datapath, spec['file']))[spec['var']]
    da = select_level(da, spec.get('level'))
    da = standardise_coords(da)
    da = select_box(da, spec.get('lat'), spec.get('lon'))
    if spec.get('resolution') is not None:
        da = regrid(da, *spec['resolution'], method=spec.get('regrid_method', 'linear'))
    return da


def merge_prep(prep_cfg=None):
    """DEFAULT_PREP updated with the (possibly partial) user config, without modifying either.

    Field specs are merged key by key, so {'x1': {'months': 'SON'}} only changes x1's months.
    `covariates` is replaced as a whole if given.
    """
    out = copy.deepcopy(DEFAULT_PREP)
    for key, val in (prep_cfg or {}).items():
        if key in ('x1', 'x2') and isinstance(val, dict):
            out[key].update(copy.deepcopy(val))
        elif key == 'covariates' and isinstance(val, dict):
            out[key] = [copy.deepcopy(val)]
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_and_preprocess(datapath, prep_cfg=None, verbose=True):
    """Raw monthly files -> standardised, flattened train/test arrays (a DataBundle).

    datapath: folder with the netCDF files.
    prep_cfg: preprocessing config (see module docstring); missing keys fall back to
              DEFAULT_PREP. For backward compatibility a float is read as test_frac.

    Steps, for every field: select level -> 0-360 lon, ascending lat -> lat/lon box ->
    optional regrid -> seasonal mean over the chosen months. Covariates are then averaged
    over their box (cos-lat weighted). All fields are aligned on season_year (only seasons
    present in every field are kept) and split contiguously: the last `test_frac` seasons
    are the test set. Each grid cell / covariate is standardised with the mean and std of
    the training seasons, and x1/x2 are then multiplied by their latitude weight.
    """
    if isinstance(prep_cfg, (int, float)):
        prep_cfg = dict(test_frac=float(prep_cfg))
    cfg = merge_prep(prep_cfg)
    cov_specs = cfg['covariates']
    if isinstance(cov_specs, dict):
        cov_specs = [cov_specs]
    cov_names = [c.get('name', f'u{i}') for i, c in enumerate(cov_specs)]

    # ── seasonal means ─────────────────────────────────────────────────────
    x1_monthly, x2_monthly = load_field(datapath, cfg['x1']), load_field(datapath, cfg['x2'])
    units = {'x1': x1_monthly.attrs.get('units', ''), 'x2': x2_monthly.attrs.get('units', '')}
    x1_raw = seasonal_mean(x1_monthly, cfg['x1']['months']).load()
    x2_raw = seasonal_mean(x2_monthly, cfg['x2']['months']).load()
    covs, cov_units = [], []
    for spec in cov_specs:
        monthly = load_field(datapath, spec)
        cov_units.append(monthly.attrs.get('units', ''))
        da = area_mean(monthly)                         # box mean first: much less data
        covs.append(seasonal_mean(da, spec['months']).load())

    # ── align on season_year only (xr.align would also intersect the different grids) ──
    common = x1_raw['season_year'].values
    for da in [x2_raw, *covs]:
        common = np.intersect1d(common, da['season_year'].values)
    if common.size == 0:
        raise ValueError('no season present in all fields — check months / files')
    x1_raw, x2_raw = x1_raw.sel(season_year=common), x2_raw.sel(season_year=common)
    u_raw = xr.concat([c.sel(season_year=common) for c in covs], dim='covariate')
    u_raw = u_raw.assign_coords(covariate=cov_names).transpose('season_year', 'covariate')

    # ── contiguous train/test split ────────────────────────────────────────
    n_seasons = common.size
    n_train = n_seasons - int(round(cfg['test_frac'] * n_seasons))
    tr = dict(season_year=slice(None, n_train))
    te = dict(season_year=slice(n_train, None))

    def norm(da, weighting=None):
        ref = da.isel(**tr)
        out = (da - ref.mean('season_year')) / ref.std('season_year')
        return out * lat_weight(da['lat'], weighting) if weighting else out

    x1_n = norm(x1_raw, cfg['x1'].get('weighting'))
    x2_n = norm(x2_raw, cfg['x2'].get('weighting'))
    u_n = norm(u_raw)

    def grid_for(name, raw, normed):
        # a cell is used if it is finite in every training season and not constant
        std = raw.isel(**tr).std('season_year').transpose('lat', 'lon')
        ok = normed.isel(**tr).notnull().all('season_year') & (std > 0)
        return FieldGrid(name, raw['lat'], raw['lon'], ok.transpose('lat', 'lon').values,
                         cfg[name].get('weighting'), std=std, units=units[name])

    g1, g2 = grid_for('x1', x1_raw, x1_n), grid_for('x2', x2_raw, x2_n)

    def flat(da, sel, grid):
        a = da.isel(**sel).transpose('season_year', 'lat', 'lon').values
        return a.reshape(a.shape[0], -1)[:, grid.mask].astype('float32')

    d = DataBundle(
        prep=cfg, covariate_names=cov_names, covariate_units=cov_units, n_train=n_train,
        x1=g1, x2=g2,
        x1_raw=x1_raw, x2_raw=x2_raw, u_raw=u_raw,
        x1_n=x1_n, x2_n=x2_n, u_n=u_n,
        x1_train=flat(x1_n, tr, g1), x1_test=flat(x1_n, te, g1),
        x2_train=flat(x2_n, tr, g2), x2_test=flat(x2_n, te, g2),
        u_train=u_n.isel(**tr).values.astype('float32'),
        u_test=u_n.isel(**te).values.astype('float32'),
        years_train=common[:n_train], years_test=common[n_train:],
    )
    for k in ('x1_train', 'x2_train', 'u_train', 'x1_test', 'x2_test', 'u_test'):
        assert d[k].size > 0 and np.isfinite(d[k]).all(), f'{k} is empty or has NaN/inf'

    if verbose:
        print(f'x1: {cfg["x1"]["var"]} months {parse_months(cfg["x1"]["months"])}, {g1}')
        print(f'x2: {cfg["x2"]["var"]} months {parse_months(cfg["x2"]["months"])}, {g2}')
        for n, s in zip(cov_names, cov_specs):
            print(f'u : {n} = {s["var"]} months {parse_months(s["months"])}, '
                  f'lat {s.get("lat")}, lon {s.get("lon")} (area mean)')
        print(f'train {d.x1_train.shape[0]} | test {d.x1_test.shape[0]} seasons '
              f'({common[0]}-{common[-1]}) | x1 dim {g1.n_features} | x2 dim {g2.n_features} '
              f'| {len(cov_names)} covariate(s)')
    return d
