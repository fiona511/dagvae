"""Plots for the DAG-VAE: per-seed diagnostics, across-seed summaries and map helpers.

All functions return a matplotlib Figure (or save/show it via `finish`) and take the
config (`cfg`) and preprocessed data (`data`, a preprocessing.DataBundle) explicitly.

Map conventions
    * Fields come out of the network standardised (and latitude-weighted). With
      cfg['plot_physical_units'] = True (default) maps are converted back to physical
      anomalies (weighting undone, x training std); otherwise only the weighting is
      undone and maps are in units of standard deviations.
    * Projections are cfg['x1_projection'] / cfg['x2_projection'], or chosen automatically
      from the lat/lon box when these are None (polar stereographic for zonally global
      high-latitude boxes, PlateCarree otherwise).
"""
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.util import add_cyclic_point

from training import acc, corr_cols, match_latents, traverse, u_sweep

# cartopy/shapely emit harmless "invalid value encountered" warnings for every contour polygon
warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely')


# ═════════════════════════════════════════════════════════════════════════════
# General helpers
# ═════════════════════════════════════════════════════════════════════════════
def finish(fig, path=None, show=True, dpi=130):
    """Save `fig` to `path` (if given), then show it or close it."""
    if path:
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def projection_for(cfg, grid):
    """Projection for field `grid` ('x1' or 'x2' FieldGrid): from cfg, else automatic."""
    proj = cfg.get(f'{grid.name}_projection')
    if proj is not None:
        return proj
    lat, lon = grid.lat.values, grid.lon.values
    zonally_global = np.ptp(lon) > 300
    if zonally_global and lat.max() <= -20:
        return ccrs.SouthPolarStereo()
    if zonally_global and lat.min() >= 20:
        return ccrs.NorthPolarStereo()
    centre = 180 if (zonally_global or lon.min() <= 180 <= lon.max()) else (lon.min() + lon.max()) / 2
    return ccrs.PlateCarree(central_longitude=float(centre))


def field_for_plot(grid, da, cfg):
    """Network-space map -> map to plot (physical anomaly or std units, see module doc)."""
    return grid.to_physical(da) if cfg.get('plot_physical_units', True) else grid.unweight(da)


def unit_for(cfg, grid):
    """Colour-bar label matching field_for_plot."""
    if not cfg.get('plot_physical_units', True):
        return '[std]'
    return cfg.get(f'{grid.name}_unit') or (f'[{grid.units}]' if grid.units else '')


def _is_polar(proj):
    return isinstance(proj, (ccrs.SouthPolarStereo, ccrs.NorthPolarStereo))


def _prep_map_ax(ax, da, proj):
    """Circular boundary cut at the data's latitude edge for polar plots; coastlines."""
    if _is_polar(proj):
        lat = da['lat'].values
        if isinstance(proj, ccrs.SouthPolarStereo):
            ax.set_extent([-180, 180, -90, lat.max()], ccrs.PlateCarree())
        else:
            ax.set_extent([-180, 180, lat.min(), 90], ccrs.PlateCarree())
        theta = np.linspace(0, 2 * np.pi, 100)
        circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5)
        ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=0.5)


def _contourf(ax, da, levels, cmap):
    vals, lon = da.values, da['lon'].values
    if np.ptp(lon) > 340:          # close the seam for global longitude grids
        vals, lon = add_cyclic_point(vals, coord=lon)
    return ax.contourf(lon, da['lat'].values, vals, levels=levels, cmap=cmap,
                       extend='both', transform=ccrs.PlateCarree())


def plot_map_grid(fields, proj, vmax, unit, row_titles=None, col_titles=None,
                  suptitle=None, cmap='RdBu_r', n_levels=21):
    """Grid of maps with one shared, symmetric colour scale.

    fields: list (rows) of lists (cols) of 2-D DataArrays (lat, lon).
    """
    nr, nc = len(fields), len(fields[0])
    w = 2.3 if _is_polar(proj) else 3.0
    fig, axes = plt.subplots(nr, nc, figsize=(w * nc + 1.2, 2.1 * nr + 0.8),
                             subplot_kw=dict(projection=proj), squeeze=False)
    levels = np.linspace(-vmax, vmax, n_levels)
    cs = None
    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            _prep_map_ax(ax, fields[i][j], proj)
            cs = _contourf(ax, fields[i][j], levels, cmap)
            if i == 0 and col_titles is not None:
                ax.set_title(col_titles[j], fontsize=9)
            if j == 0 and row_titles is not None:
                ax.text(-0.08, 0.5, row_titles[i], transform=ax.transAxes, rotation=90,
                        va='center', ha='right', fontsize=10)
    cb = fig.colorbar(cs, ax=axes.ravel().tolist(), shrink=min(0.8, 0.25 + 0.12 * nr), pad=0.02)
    cb.set_label(unit)
    if suptitle:
        fig.suptitle(suptitle, y=1.0)
    return fig


def _robust_vmax(rows):
    """99th percentile of |values| over all maps (shared colour scale)."""
    allv = np.concatenate([np.abs(r.values).ravel() for row in rows for r in row])
    return float(np.nanpercentile(allv, 99)) or 1.0


# old name
_traversal_vmax = _robust_vmax


def _get(obj, key):
    """Read `key` from a dict or an attribute (SkillMonitor or its saved dict)."""
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


# ═════════════════════════════════════════════════════════════════════════════
# Per-seed plots
# ═════════════════════════════════════════════════════════════════════════════
def plot_seed_figures(seed, out, show, cfg, data, hist, monitor, enc_tr, enc_te, kl1, kl2,
                      decoder, pred_model):
    """All per-seed figures, saved to `out` (called from training.run_seed)."""
    finish(plot_history(hist, monitor, seed), f'{out}/training_curves.png', show)
    finish(plot_latent_timeseries(enc_tr, enc_te, seed, cfg, data), f'{out}/latent_timeseries.png', show)
    finish(plot_latent_diagnostics(pred_model, kl1, kl2, seed, data.covariate_names),
           f'{out}/latent_diagnostics.png', show)
    finish(plot_acc_maps(enc_te, seed, cfg, data), f'{out}/acc_maps_test.png', show)
    finish(plot_traversals(decoder, enc_tr, 'z1', seed, cfg, data), f'{out}/traversal_z1.png', show)
    finish(plot_traversals(decoder, enc_tr, 'z2', seed, cfg, data), f'{out}/traversal_z2.png', show)
    finish(plot_traversal_z1_to_x2(decoder, pred_model, enc_tr, seed, cfg, data),
           f'{out}/traversal_z1_to_x2.png', show)
    finish(plot_traversal_u_to_x2(decoder, pred_model, enc_tr, seed, cfg, data),
           f'{out}/traversal_u_to_x2.png', show)


def plot_history(hist, monitor, seed):
    """Loss terms (train and test) per epoch, plus the test ACC tracked by SkillMonitor."""
    keys = [('loss', 'total loss'), ('reconstruction_loss_z1', 'rec loss x1'),
            ('reconstruction_loss_z2', 'rec loss x2'), ('regularization_loss_z1', 'KL z1'),
            ('regularization_loss_z2', 'KL z2 | p(z2|z1,u)'),
            ('prediction_loss_x2', 'pred loss x2 (decoded p(z2|z1,u))')]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    axes.flat[7].axis('off')
    for ax, (k, title) in zip(axes.flat, keys):
        ax.plot(hist[k], label='train', lw=1.2)
        if 'val_' + k in hist:
            ax.plot(hist['val_' + k], label='test', lw=1.2, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel('epoch')
        if np.nanmin(hist[k]) > 0:
            ax.set_yscale('log')
        ax.grid(alpha=0.3)
    axes.flat[0].legend()
    ax = axes.flat[6]
    epochs, acc_pred, acc_rec = (_get(monitor, k) for k in ('epochs', 'acc_pred', 'acc_rec'))
    ax.plot(epochs, acc_pred, label='x2 pred (z1 -> x2)')
    ax.plot(epochs, acc_rec, label='x2 reconstruction', alpha=0.8)
    ax.set_title('test ACC (area-weighted)')
    ax.set_xlabel('epoch')
    ax.set_ylim(min(0, np.nanmin(acc_pred) - 0.05), 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle(f'seed {seed} — training curves')
    fig.tight_layout()
    return fig


def _combine_splits(a_tr, a_te, data):
    """Concatenate train and test arrays in season order; also return an is-test mask."""
    yrs = np.concatenate([data.years_train, data.years_test])
    split = np.r_[np.zeros(len(data.years_train), bool), np.ones(len(data.years_test), bool)]
    a = np.concatenate([a_tr, a_te])
    o = np.argsort(yrs)
    return yrs[o], a[o], split[o]


def plot_latent_timeseries(enc_tr, enc_te, seed, cfg, data):
    """Left: z1 posterior means (±2σ). Right: encoded z2 vs its prediction from z1 (+u)."""
    d1, d2 = cfg['latent_dim_z1'], cfg['latent_dim_z2']
    n = max(d1, d2)
    fig, axes = plt.subplots(n, 2, figsize=(15, 1.9 * n + 0.6), sharex=True, squeeze=False)
    yrs, _, is_te = _combine_splits(enc_tr['z1_mean'], enc_te['z1_mean'], data)
    for k in range(n):
        for col, (name, d) in enumerate([('z1', d1), ('z2', d2)]):
            ax = axes[k, col]
            if k >= d:
                ax.axis('off')
                continue
            _, m, _ = _combine_splits(enc_tr[f'{name}_mean'][:, k], enc_te[f'{name}_mean'][:, k], data)
            _, lv, _ = _combine_splits(enc_tr[f'{name}_log_var'][:, k], enc_te[f'{name}_log_var'][:, k], data)
            sd = np.exp(0.5 * lv)
            ax.fill_between(yrs, m - 2 * sd, m + 2 * sd, color='0.8', lw=0, label='±2σ posterior')
            ax.plot(yrs, m, color='k', lw=1, label=f'{name} mean (encoded)')
            ax.scatter(yrs[is_te], m[is_te], s=12, color='k', zorder=3)
            title = f'{name}[{k}]'
            if name == 'z2':
                _, p, _ = _combine_splits(enc_tr['pz2'][:, k], enc_te['pz2'][:, k], data)
                ax.plot(yrs, p, color='C3', lw=1.2, label='predicted from z1, u')
                r_tr = corr_cols(enc_tr['z2_mean'][:, [k]], enc_tr['pz2'][:, [k]])[0]
                r_te = corr_cols(enc_te['z2_mean'][:, [k]], enc_te['pz2'][:, [k]])[0]
                title += f'   r(train)={r_tr:.2f}  r(test)={r_te:.2f}'
            ax.set_title(title, fontsize=9, loc='left')
            ax.grid(alpha=0.3)
            if k == 0:
                ax.legend(fontsize=7, loc='upper right', ncol=3)
    for ax in axes[-1]:
        ax.set_xlabel('season year  (dots = test years)')
    fig.suptitle(f'seed {seed} — latent time series', y=1.0)
    fig.tight_layout()
    return fig


def plot_latent_diagnostics(pred_model, kl1_tr, kl2_tr, seed, covariate_names=None):
    """Weights of p(z2|z1,u) as a matrix, and KL per latent dim (grey = inactive)."""
    W = pred_model.get_layer('pz2_z1').get_weights()[0]            # [d1, d2]
    Wu, b = pred_model.get_layer('pz2_u').get_weights()             # [n_cov, d2], [d2]
    M = np.vstack([W, Wu, b[None]])
    cov_labels = covariate_names or [f'u[{i}]' for i in range(Wu.shape[0])]
    labels = [f'z1[{i}]' for i in range(W.shape[0])] + list(cov_labels) + ['bias']
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.45 * len(labels) + 2),
                             gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    ax = axes[0]
    vmax = np.abs(M).max() or 1
    im = ax.imshow(M, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(M.shape[1]), [f'z2[{j}]' for j in range(M.shape[1])])
    ax.axhline(W.shape[0] - 0.5, color='k', lw=0.8)
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                color='w' if abs(v) > 0.6 * vmax else 'k')
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('p(z2 | z1, u) weights')
    for ax, kl, name in [(axes[1], kl1_tr, 'z1  (vs N(0,1))'), (axes[2], kl2_tr, 'z2  (vs p(z2|z1,u))')]:
        ax.bar(range(len(kl)), kl, color=['C0' if v > 0.01 else '0.7' for v in kl])
        ax.axhline(0.01, color='k', lw=0.6, ls='--')
        ax.set_title(f'KL per dim, train — {name}')
        ax.set_xlabel('latent dim')
        ax.set_xticks(range(len(kl)))
        ax.grid(alpha=0.3, axis='y')
    fig.suptitle(f'seed {seed} — latent diagnostics (grey bars = inactive / collapsed)')
    fig.tight_layout()
    return fig


def plot_acc_maps(enc_te, seed, cfg, data):
    """Test-set gridpoint ACC of x1 / x2 reconstruction and of the x2 prediction."""
    g1, g2, yrs = data.x1, data.x2, data.years_test
    obs1, obs2 = g1.to_grid(data.x1_test, yrs), g2.to_grid(data.x2_test, yrs)
    maps = [(acc(obs1, g1.to_grid(enc_te['x1_rec'], yrs)), g1, 'x1 reconstruction'),
            (acc(obs2, g2.to_grid(enc_te['x2_rec'], yrs)), g2, 'x2 reconstruction'),
            (acc(obs2, g2.to_grid(enc_te['x2_pred'], yrs)), g2, 'x2 predicted from z1, u')]
    fig = plt.figure(figsize=(16, 4.6))
    levels = np.linspace(-1, 1, 21)
    cs = None
    for i, (da, grid, title) in enumerate(maps):
        proj = projection_for(cfg, grid)
        ax = fig.add_subplot(1, 3, i + 1, projection=proj)
        _prep_map_ax(ax, da, proj)
        cs = _contourf(ax, da, levels, 'RdBu_r')
        mean_acc = float(da.weighted(grid.coslat).mean(('lat', 'lon')))
        ax.set_title(f'{title}\nmean ACC = {mean_acc:.3f}', fontsize=10)
    fig.colorbar(cs, ax=fig.axes, shrink=0.7, pad=0.02, label='ACC')
    fig.suptitle(f'seed {seed} — test-set ACC maps', y=1.02)
    return fig


def plot_traversals(decoder, enc_tr, which, seed, cfg, data):
    """Sweep each dim of z1 (-> x1 patterns) or z2 (-> x2 patterns) from its min to max
    over the training set, other dims at their mean, and decode."""
    n = cfg['traversal_n']
    z1ref, z2ref = enc_tr['z1_mean'], enc_tr['z2_mean']
    zref = z1ref if which == 'z1' else z2ref
    other = np.tile((z2ref if which == 'z1' else z1ref).mean(0), (n, 1)).astype('float32')
    out_idx = 0 if which == 'z1' else 1
    grid = data.x1 if which == 'z1' else data.x2
    base = decoder.predict([z1ref.mean(0, keepdims=True), z2ref.mean(0, keepdims=True)],
                           verbose=0)[out_idx]
    rows, row_titles = [], []
    for k in range(zref.shape[1]):
        s = traverse(zref, k, n)
        dec = decoder.predict([s, other] if which == 'z1' else [other, s], verbose=0)[out_idx]
        if cfg['traversal_relative']:
            dec = dec - base
        rec = field_for_plot(grid, grid.to_grid(dec), cfg)
        rows.append([rec.isel(season_year=i) for i in range(n)])
        row_titles.append(f'{which}[{k}]\n[{s[0, k]:+.1f} … {s[-1, k]:+.1f}]')
    rel = ' (minus decoded latent mean)' if cfg['traversal_relative'] else ''
    return plot_map_grid(rows, projection_for(cfg, grid), _robust_vmax(rows), unit_for(cfg, grid),
                         row_titles=row_titles, col_titles=[f'step {i + 1}' for i in range(n)],
                         suptitle=f'seed {seed} — {which} traversals → {grid.name}{rel}')


def plot_traversal_z1_to_x2(decoder, pred_model, enc_tr, seed, cfg, data):
    """Vary one z1 dim (others at their mean, u at its training mean), map it through
    p(z2 | z1, u) and decode x2: the x2 pattern the model predicts for each z1 direction."""
    n = cfg['traversal_n']
    g2 = data.x2
    z1ref = enc_tr['z1_mean']
    u_ref = np.tile(data.u_train.mean(0), (n, 1)).astype('float32')
    pz2_base = pred_model.predict([z1ref.mean(0, keepdims=True), u_ref[:1]], verbose=0)
    base = decoder.predict([z1ref.mean(0, keepdims=True), pz2_base], verbose=0)[1]
    rows, row_titles = [], []
    for k in range(z1ref.shape[1]):
        s = traverse(z1ref, k, n)
        pz2 = pred_model.predict([s, u_ref], verbose=0)
        dec = decoder.predict([s, pz2], verbose=0)[1]
        if cfg['traversal_relative']:
            dec = dec - base
        rec = field_for_plot(g2, g2.to_grid(dec), cfg)
        rows.append([rec.isel(season_year=i) for i in range(n)])
        row_titles.append(f'z1[{k}]\n[{s[0, k]:+.1f} … {s[-1, k]:+.1f}]')
    rel = ' (minus decoded latent mean)' if cfg['traversal_relative'] else ''
    return plot_map_grid(rows, projection_for(cfg, g2), _robust_vmax(rows), unit_for(cfg, g2),
                         row_titles=row_titles, col_titles=[f'step {i + 1}' for i in range(n)],
                         suptitle=f'seed {seed} — z1 traversals → p(z2|z1,u) → x2{rel}')


# ═════════════════════════════════════════════════════════════════════════════
# Across-seed summary plots (saved to out_dir and shown)
# ═════════════════════════════════════════════════════════════════════════════
def plot_summary_metrics(df, ref, cfg, show=True):
    """Spread of key test metrics across seeds, with ensemble and linear baseline."""
    panels = [('test_acc_x2_pred', 'ACC x2 pred (z1→x2), test'), ('test_acc_x2_rec', 'ACC x2 rec, test'),
              ('test_acc_x1_rec', 'ACC x1 rec, test'), ('test_pcorr_x2_pred', 'pattern corr x2 pred, test'),
              ('test_loss_total', 'total loss, test'), ('test_loss_kl_z1', 'KL z1, test'),
              ('test_loss_kl_z2', 'KL z2, test'), ('test_z2_pred_r_mean', 'r(z2, pz2) mean, test')]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (col, title) in zip(axes.flat, panels):
        ax.scatter(np.zeros(len(df)), df[col], alpha=0.7)
        for s, v in df[col].items():
            ax.annotate(str(s), (0, v), xytext=(6, 0), textcoords='offset points', fontsize=7, va='center')
        ax.axhline(df[col].mean(), color='k', lw=1, label='mean over seeds')
        if col == 'test_acc_x2_pred':
            ax.axhline(ref['acc_ens'], color='C3', lw=1.2, ls='--', label='seed-ensemble mean')
            ax.axhline(ref['acc_base'], color='C2', lw=1.2, ls=':', label='linear baseline')
            ax.legend(fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.grid(alpha=0.3, axis='y')
    fig.suptitle('Spread across seeds (numbers = seed)')
    fig.tight_layout()
    finish(fig, os.path.join(cfg['out_dir'], 'summary_metrics.png'), show)


def plot_summary_training(results, ref, cfg, show=True):
    """Test loss and test ACC of the x2 prediction during training, all seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for s, r in results.items():
        axes[0].plot(r['hist']['val_loss'], lw=0.9, label=f'seed {s}')
        axes[1].plot(r['monitor']['epochs'], r['monitor']['acc_pred'], lw=0.9)
    axes[0].set_yscale('log')
    axes[0].set_title('test loss')
    axes[1].set_title('test ACC of x2 prediction from z1')
    axes[1].axhline(ref['acc_base'], color='k', ls=':', lw=1, label='linear baseline')
    for ax in axes:
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    finish(fig, os.path.join(cfg['out_dir'], 'summary_training.png'), show)


def plot_summary_acc_maps(ref, cfg, data, show=True):
    """Test ACC maps: seed-ensemble VAE prediction vs linear baseline."""
    g2, yrs = data.x2, data.years_test
    obs2 = g2.to_grid(data.x2_test, yrs)
    fig = plot_map_grid([[acc(obs2, g2.to_grid(ref['x2_pred_ens'], yrs)),
                          acc(obs2, g2.to_grid(ref['x2_pred_base'], yrs))]],
                        projection_for(cfg, g2), 1.0, 'ACC',
                        col_titles=[f'seed-ensemble VAE prediction ({ref["acc_ens"]:.3f})',
                                    f'linear PCA regression ({ref["acc_base"]:.3f})'],
                        suptitle='Test ACC of x2 prediction from x1')
    finish(fig, os.path.join(cfg['out_dir'], 'summary_acc_maps.png'), show)


def plot_latent_stability(results, cfg, show=True):
    """Pairwise mean matched |r| of latent time series between seeds."""
    seeds = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = None
    for ax, name in zip(axes, ['z1', 'z2']):
        S = np.ones((len(seeds), len(seeds)))
        for i, a in enumerate(seeds):
            for j, b in enumerate(seeds):
                if i != j:
                    S[i, j] = match_latents(results[a]['enc_tr'][f'{name}_mean'],
                                            results[b]['enc_tr'][f'{name}_mean'])[2].mean()
        im = ax.imshow(S, vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(len(seeds)), seeds)
        ax.set_yticks(range(len(seeds)), seeds)
        off = S[~np.eye(len(seeds), dtype=bool)]
        ax.set_title(f'{name}: mean matched |r| between seeds\n'
                     f'(off-diagonal mean = {off.mean() if off.size else np.nan:.2f})')
        ax.set_xlabel('seed')
        ax.set_ylabel('seed')
    fig.colorbar(im, ax=axes, shrink=0.8)
    finish(fig, os.path.join(cfg['out_dir'], 'summary_latent_stability.png'), show)


def plot_aligned_latents(results, cfg, data, show=True):
    """Every seed's latent dims matched (and sign-flipped) to the first seed, overlaid."""
    seeds = list(results)
    ref = seeds[0]
    years = data.years_train
    o = np.argsort(years)
    for name in ['z1', 'z2']:
        d = cfg[f'latent_dim_{name}']
        zref = results[ref]['enc_tr'][f'{name}_mean']
        fig, axes = plt.subplots(d, 1, figsize=(13, 1.8 * d + 0.6), sharex=True, squeeze=False)
        matched = {s: match_latents(zref, results[s]['enc_tr'][f'{name}_mean']) for s in seeds}
        for s in seeds:
            z = results[s]['enc_tr'][f'{name}_mean']
            perm, sign, _ = matched[s]
            for k in range(d):
                axes[k, 0].plot(years[o], sign[k] * z[o, perm[k]],
                                lw=2 if s == ref else 0.8, color='k' if s == ref else None,
                                alpha=1 if s == ref else 0.6)
        for k in range(d):
            rs = [matched[s][2][k] for s in seeds if s != ref]
            axes[k, 0].set_title(f'{name}[{k}] of seed {ref} (black) and matched dims of other seeds'
                                 + (f' — median |r| = {np.median(rs):.2f}' if rs else ''),
                                 fontsize=9, loc='left')
            axes[k, 0].grid(alpha=0.3)
        axes[-1, 0].set_xlabel('season year (train)')
        fig.tight_layout()
        finish(fig, os.path.join(cfg['out_dir'], f'summary_aligned_{name}.png'), show)


def _step_labels(steps):
    """Column titles and per-row range labels for a sweep given in std units [n_rows, n].

    If every row uses the same steps (the 'std' sweep) the columns are labelled in σ;
    otherwise (the 'minmax' sweep) columns are 'step i' and each row gets its range."""
    steps = np.atleast_2d(steps)
    if np.allclose(steps, steps[:1]):
        return [f'{v:+.1f} σ' for v in steps[0]], [''] * len(steps)
    return ([f'step {i + 1}' for i in range(steps.shape[1])],
            [f'\n[{r[0]:+.1f}σ … {r[-1]:+.1f}σ]' for r in steps])


def _sweep_text(sweep):
    return 'training min → max' if sweep == 'minmax' else 'mean ± n σ'


def plot_seed_mean_z1_response(resp, cfg, data, agree_threshold=0.8, show=True,
                               filename='summary_traversal_z1_to_x2_seedmean.png'):
    """Seed-mean x2 response to each z1 direction (output of
    training.seed_mean_z1_response), stippled where >= agree_threshold of the seeds agree
    on the sign. With the 'minmax' sweep, row labels give the range of the reference seed."""
    match_r, d1 = resp['match_r'], resp['ens'].shape[0]
    col_titles, ranges = _step_labels(resp['steps'])
    row_titles = [f'z1[{k}] (seed {resp["ref_seed"]})'
                  + (f'\nmatch |r| {np.median(match_r[1:, k]):.2f}' if len(resp['seeds']) > 1 else '')
                  + ranges[k] for k in range(d1)]
    fig = plot_x2_response_grid(
        resp['ens'], cfg, data, row_titles, col_titles,
        f'z1 → p(z2|z1,u) → x2 response ({_sweep_text(resp.get("sweep"))}), mean of '
        f'{len(resp["seeds"])} seeds (stippling: ≥{agree_threshold:.0%} sign agreement)',
        agree=resp['agree'], agree_threshold=agree_threshold)
    finish(fig, os.path.join(cfg['out_dir'], filename), show)
    return fig


def plot_x2_response_grid(fields, cfg, data, row_titles, col_titles, suptitle,
                          agree=None, agree_threshold=0.8):
    """Rows of x2 response maps, optionally stippled where agree >= agree_threshold.

    fields, agree: [n_rows, n_steps, x2 features] in network space (as returned by
    training.seed_mean_z1_response / covariate_response / u_sweep).
    """
    g2 = data.x2
    n_rows, n = fields.shape[:2]
    rows = [[field_for_plot(g2, g2.to_grid(fields[k, c]), cfg).isel(season_year=0) for c in range(n)]
            for k in range(n_rows)]
    fig = plot_map_grid(rows, projection_for(cfg, g2), _robust_vmax(rows), unit_for(cfg, g2),
                        row_titles=row_titles, col_titles=col_titles, suptitle=suptitle)
    if agree is not None:
        for idx, ax in enumerate(fig.axes[:n_rows * n]):
            k, c = divmod(idx, n)
            a = g2.to_grid(agree[k, c]).isel(season_year=0)
            if np.nanmax(a.values, initial=0) >= agree_threshold:
                ax.contourf(a['lon'], a['lat'], a.values, levels=[agree_threshold, 1.01],
                            hatches=['...'], colors='none', transform=ccrs.PlateCarree())
    return fig


def _u_row_titles(names, u_std=None, units=None, ranges=None):
    titles = []
    for k, name in enumerate(names):
        t = f'u: {name}'
        if u_std is not None:
            t += f'\n1σ = {u_std[k]:.3g}' + (f' {units[k]}' if units and units[k] else '')
        titles.append(t + (ranges[k] if ranges else ''))
    return titles


def _u_std_physical(data):
    return data.u_raw.isel(season_year=slice(None, data.n_train)).std('season_year').values


def plot_traversal_u_to_x2(decoder, pred_model, enc_tr, seed, cfg, data, sigma=2.0, sweep=None):
    """One seed: sweep each covariate (z1 at its training mean), map through p(z2|z1,u)
    and decode x2, relative to u at its mean (training.u_sweep).

    sweep: 'std' (mean ± sigma std), 'minmax' (training min ... max) or None for
    cfg['response_sweep']."""
    sweep = cfg.get('response_sweep', 'std') if sweep is None else sweep
    fields, steps = u_sweep(decoder, pred_model, enc_tr['z1_mean'].mean(0), data,
                            n=cfg['traversal_n'], sweep=sweep, sigma=sigma)
    col_titles, ranges = _step_labels(steps)
    return plot_x2_response_grid(
        fields, cfg, data,
        _u_row_titles(data.covariate_names, _u_std_physical(data),
                      getattr(data, 'covariate_units', None), ranges),
        col_titles,
        f'seed {seed} — u traversals ({_sweep_text(sweep)}) → p(z2|z1,u) → x2 '
        f'(z1 at its mean, minus u = mean)')


def plot_covariate_response(resp, cfg, data, agree_threshold=0.8, show=True,
                            filename='summary_traversal_u_to_x2_seedmean.png'):
    """Seed-mean x2 response to each covariate (output of training.covariate_response),
    stippled where >= agree_threshold of the seeds agree on the sign."""
    col_titles, ranges = _step_labels(resp['steps'])
    fig = plot_x2_response_grid(
        resp['ens'], cfg, data,
        _u_row_titles(resp['names'], resp['u_std'], resp.get('units'), ranges), col_titles,
        f'u → p(z2|z1,u) → x2 response ({_sweep_text(resp.get("sweep"))}), mean of '
        f'{len(resp["seeds"])} seeds (stippling: ≥{agree_threshold:.0%} sign agreement)',
        agree=resp['agree'], agree_threshold=agree_threshold)
    finish(fig, os.path.join(cfg['out_dir'], filename), show)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Older general-purpose map plots (from support_functions_0926)
# ═════════════════════════════════════════════════════════════════════════════
def visualise_contourplot_labels(cluster_centers, regime_names, vmin, vmax, steps, color_scheme,
                                 labels, col_number=5, borders=True,
                                 projection=ccrs.PlateCarree(central_longitude=180)):
    """One map per cluster centre, titled with the regime name and its frequency.

    cluster_centers: DataArray (cluster, lat, lon); labels: cluster index of every sample.
    """
    nt = cluster_centers.shape[0]
    x, y = np.meshgrid(cluster_centers.lon, cluster_centers.lat)
    fig, axes = plt.subplots(1, col_number, figsize=(14, 5), subplot_kw=dict(projection=projection))
    for i in range(nt):
        ax = axes.flat[i]
        ax.contourf(x, y, cluster_centers[i, :, :], levels=np.arange(vmin, vmax, steps),
                    transform=ccrs.PlateCarree(), cmap=color_scheme)
        ax.coastlines()
        if borders:
            ax.add_feature(cartopy.feature.BORDERS)
        ax.set_title('{}, {:4.1f}%'.format(regime_names[i], 100 * np.mean(labels == i)))
    plt.tight_layout()
    return fig


def visualise_contourplot_latent_legend(samples, vmin, vmax, steps, color_scheme,
                                        col_number=5, borders=True,
                                        projection=ccrs.PlateCarree(central_longitude=180),
                                        unit_name='[mm/day]', shrink_value=0.4):
    """One map per sample (sample, lat, lon) in a row, with a shared colour bar."""
    nt = samples.shape[0]
    x, y = np.meshgrid(samples.lon, samples.lat)
    fig, axes = plt.subplots(1, col_number, figsize=(1.5 + 3 * col_number, 5),
                             subplot_kw=dict(projection=projection))
    cs = None
    for i in range(nt):
        ax = axes.flat[i]
        cs = ax.contourf(x, y, samples[i, :, :], levels=np.arange(vmin, vmax + steps, steps),
                         transform=ccrs.PlateCarree(), cmap=color_scheme)
        ax.coastlines()
        if borders:
            ax.add_feature(cartopy.feature.BORDERS)
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    cbar = fig.colorbar(cs, ax=axes.ravel().tolist(), orientation='vertical',
                        shrink=shrink_value, pad=0.02)
    cbar.set_label(unit_name)
    return fig
