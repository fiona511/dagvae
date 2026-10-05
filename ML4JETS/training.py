"""Training support for the DAG-VAE: config, multi-seed training, metrics and baselines.

Typical use (see the notebook):

    import preprocessing as pp, training as tr, visualisation as vis
    data = pp.load_and_preprocess(datapath, PREP)
    cfg = tr.make_config(CFG)                      # fills in defaults for missing keys
    results = tr.run_all_seeds(cfg, data)          # {seed: result dict}
    df = tr.metrics_table(results, cfg)
    ref = tr.baseline_and_ensemble(results, df, cfg, data)

Every function takes the config (`cfg`) and the preprocessed data (`data`, a
preprocessing.DataBundle) explicitly — there is no hidden module state, so several
configurations / datasets can be used side by side in one session.

For every seed, run_seed
   1. builds fresh encoder / decoder / prediction model (seeded *before* building, so the
      seed controls weight initialisation, dropout and sampling),
   2. trains the joint VAE and tracks the test ACC of the z1 -> x2 prediction,
   3. reports all loss terms (train/test), per-variable reconstruction skill (MSE,
      explained variance, area-weighted ACC), the ACC of the x2 prediction from z1 (+u),
      latent KL per dimension / active units, and how well p(z2|z1,u) predicts encoded z2,
   4. makes the per-seed figures (visualisation.py) and saves weights, history and latents.
"""
import copy
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from dagvae import build_encoder, build_decoder, build_prediction_model_linear, build_vae


# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_CFG = dict(
    seeds=list(range(10)),

    # architecture
    latent_dim_z1=1,
    latent_dim_z2=3,
    hidden_dims=(128, 64, 32),     # encoder widths, input side first (decoder mirrors it)
    activation='relu',
    dropout_rate=0.3,

    # loss
    reconstruction_loss_factors=[1.0, 1.0],
    kl_loss_factors=[1.0, 1.0],
    prediction_loss_factor=0.3,    # MSE(x2, decode(z1, p(z2|z1,u))); 0 = off
    regularization_term=0.01,      # L1 on z1 -> z2 weights
    covariate_reg=0.0,             # L1 on u -> z2 weights

    # optimisation
    learning_rate=1e-3,
    epochs=100,
    batch_size=128,
    deterministic_ops=False,       # tf.config.experimental.enable_op_determinism()

    # monitoring / output
    print_every=20,                # progress line every n epochs
    acc_every=5,                   # test ACC tracked every n epochs
    out_dir='runs/vae_seeds',
    make_plots=True,               # per-seed figures (always saved to disk when made)
    show_figs_for_seeds='all',     # 'all', or a list of seeds whose figures are displayed

    # plotting
    traversal_n=7,                 # steps per latent traversal
    traversal_relative=False,      # subtract the decoded latent mean from traversals
    response_sweep='std',          # u and seed-mean z1 response maps: 'std' (mean ± 2 std)
                                   # or 'minmax' (training min ... max)
    plot_physical_units=True,      # maps in physical units (x train std) instead of std units
    x1_unit=None,                  # colour-bar labels; None -> taken from the file attrs
    x2_unit=None,
    x1_projection=None,            # cartopy projections; None -> chosen from the lat/lon box
    x2_projection=None,
)


def make_config(cfg=None, **overrides):
    """DEFAULT_CFG updated with `cfg` and keyword overrides (inputs are not modified).

    The old keys dim_layer0/1/2 are still accepted and turned into hidden_dims.
    """
    out = copy.copy(DEFAULT_CFG)
    out.update(cfg or {})
    out.update(overrides)
    if 'dim_layer0' in out:
        out['hidden_dims'] = tuple(out.pop(f'dim_layer{i}') for i in range(3) if f'dim_layer{i}' in out)
    out.pop('x2_is_sqrtcos_weighted', None)        # now stored with the data (data.x2.weighting)
    return out


def _setup(cfg):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    if cfg.get('deterministic_ops'):
        tf.config.experimental.enable_op_determinism()


# ═════════════════════════════════════════════════════════════════════════════
# Metrics (numpy / xarray)
# ═════════════════════════════════════════════════════════════════════════════
def acc(obs, pred, dim='season_year'):
    """Gridpoint anomaly correlation over time (xarray in, map out)."""
    o, p = obs - obs.mean(dim), pred - pred.mean(dim)
    return (o * p).sum(dim) / np.sqrt((o**2).sum(dim) * (p**2).sum(dim))


def acc_flat(obs, pred, w):
    """Area-weighted mean of the gridpoint ACC over time, on flat [time, feature] arrays."""
    o, p = obs - obs.mean(0), pred - pred.mean(0)
    den = np.sqrt((o**2).sum(0) * (p**2).sum(0))
    a = np.full(den.shape, np.nan)
    ok = den > 0
    a[ok] = (o * p).sum(0)[ok] / den[ok]
    ok &= np.isfinite(a)
    return float(np.sum(a[ok] * w[ok]) / np.sum(w[ok]))


def pattern_corr(obs, pred, w):
    """Area-weighted spatial (pattern) correlation for every season; returns [n]."""
    wn = w / w.sum()
    o = obs - (obs * wn).sum(1, keepdims=True)
    p = pred - (pred * wn).sum(1, keepdims=True)
    return (o * p * wn).sum(1) / np.sqrt((o**2 * wn).sum(1) * (p**2 * wn).sum(1))


def explained_variance(obs, pred):
    """1 - SSE/SST in the network's (standardised) space."""
    return float(1 - np.sum((obs - pred)**2) / np.sum((obs - obs.mean(0))**2))


def kl_per_dim(mean, log_var, prior_mean=0.0):
    """Mean KL(q || N(prior_mean, 1)) per latent dimension."""
    return (0.5 * ((mean - prior_mean)**2 + np.exp(log_var) - 1 - log_var)).mean(0)


def corr_cols(a, b):
    """Pearson correlation of matching columns of a and b."""
    a, b = a - a.mean(0), b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a**2).sum(0) * (b**2).sum(0))


def match_latents(ref, other):
    """Match latent dims of `other` to `ref` (Hungarian on |corr|).

    Returns perm, sign, |r|: dim k of ref corresponds to sign[k] * other[:, perm[k]].
    """
    d = ref.shape[1]
    C = np.nan_to_num(np.corrcoef(ref.T, other.T)[:d, d:])
    r, c = linear_sum_assignment(-np.abs(C))
    return c, np.sign(C[r, c]), np.abs(C[r, c])


def traverse(z, dim, n):
    """n latent vectors: dim `dim` swept from its min to max, the others at their mean."""
    out = np.tile(z.mean(0), (n, 1))
    out[:, dim] = np.linspace(z[:, dim].min(), z[:, dim].max(), n)
    return out.astype('float32')


# ═════════════════════════════════════════════════════════════════════════════
# Building and running the model
# ═════════════════════════════════════════════════════════════════════════════
class SkillMonitor(keras.callbacks.Callback):
    """Tracks the test ACC of the z1 -> x2 prediction (and of the x2 reconstruction) every
    `every` epochs, and prints a compact progress line every `print_every` epochs."""

    def __init__(self, encoder, decoder, pred_model, x1, x2, u, w2, every=10, print_every=100):
        super().__init__()
        self.enc, self.dec, self.pm = encoder, decoder, pred_model
        self.x1, self.x2, self.u, self.w2 = x1, x2, u, w2
        self.every, self.print_every = every, print_every
        self.epochs, self.acc_pred, self.acc_rec = [], [], []

    def _skill(self):
        z1m, _, _, z2m, _, _ = self.enc([self.x1, self.x2], training=False)
        pz2 = self.pm([z1m, self.u], training=False)
        _, x2_rec = self.dec([z1m, z2m], training=False)
        _, x2_pred = self.dec([z1m, pz2], training=False)
        return (acc_flat(self.x2, x2_pred.numpy(), self.w2),
                acc_flat(self.x2, x2_rec.numpy(), self.w2))

    def on_epoch_end(self, epoch, logs=None):
        e = epoch + 1
        if e % self.every == 0 or e == 1:
            a_pred, a_rec = self._skill()
            self.epochs.append(e)
            self.acc_pred.append(a_pred)
            self.acc_rec.append(a_rec)
        if self.print_every and e % self.print_every == 0:
            logs = logs or {}
            print(f'  epoch {e:5d}  loss {logs.get("loss", np.nan):9.2f}  '
                  f'val_loss {logs.get("val_loss", np.nan):9.2f}  '
                  f'test ACC z1->x2 {self.acc_pred[-1]:.3f}  x2 rec {self.acc_rec[-1]:.3f}')


def build_all(seed, cfg, data):
    """Fresh encoder, decoder, prediction model and compiled VAE for one seed."""
    K.clear_session()
    keras.utils.set_random_seed(seed)   # must come BEFORE building: controls weight init
    c = cfg
    d1, d2, n_cov = data.x1.n_features, data.x2.n_features, data.n_covariates
    encoder = build_encoder(d1, d2, c['latent_dim_z1'], c['latent_dim_z2'],
                            c['hidden_dims'], c['activation'], c['dropout_rate'])
    decoder = build_decoder(c['latent_dim_z1'], c['latent_dim_z2'], d1, d2,
                            c['hidden_dims'], c['activation'], c['dropout_rate'])
    pred_model = build_prediction_model_linear(c['latent_dim_z1'], c['latent_dim_z2'], n_cov,
                                               c['regularization_term'], c['covariate_reg'])
    vae = build_vae(encoder, decoder, pred_model, d1, d2,
                    c['reconstruction_loss_factors'], c['kl_loss_factors'],
                    n_covariates=n_cov, chosen_learning_rate=c['learning_rate'], seed=seed,
                    prediction_loss_factor=c['prediction_loss_factor'])
    return encoder, decoder, pred_model, vae


def encode_all(encoder, decoder, pred_model, x1, x2, u, batch_size=128):
    """Deterministic pass (posterior means, no dropout): latents, reconstructions and the
    x2 prediction decoded from p(z2 | z1, u)."""
    bs = batch_size
    z1m, z1lv, _, z2m, z2lv, _ = encoder.predict([x1, x2], batch_size=bs, verbose=0)
    pz2 = pred_model.predict([z1m, u], batch_size=bs, verbose=0)
    x1_rec, x2_rec = decoder.predict([z1m, z2m], batch_size=bs, verbose=0)
    _, x2_pred = decoder.predict([z1m, pz2], batch_size=bs, verbose=0)
    return dict(z1_mean=z1m, z1_log_var=z1lv, z2_mean=z2m, z2_log_var=z2lv, pz2=pz2,
                x1_rec=x1_rec, x2_rec=x2_rec, x2_pred=x2_pred)


def split_metrics(vae, enc, x1, x2, u, data, batch_size=128):
    """All scalar diagnostics for one split. Returns (metrics dict, KL per z1 dim, KL per z2 dim)."""
    w1, w2 = data.x1.area_weights, data.x2.area_weights
    losses = vae.evaluate([x1, x2, u], batch_size=batch_size, verbose=0, return_dict=True)
    m = {
        'loss_total': losses['loss'],
        'loss_rec_x1': losses['reconstruction_loss_z1'],
        'loss_rec_x2': losses['reconstruction_loss_z2'],
        'loss_kl_z1': losses['regularization_loss_z1'],
        'loss_kl_z2': losses['regularization_loss_z2'],
        'loss_pred_x2': losses['prediction_loss_x2'],
        # plain MSE / explained variance in standardised units (not scaled by loss factors)
        'mse_x1_rec': float(np.mean((x1 - enc['x1_rec'])**2)),
        'mse_x2_rec': float(np.mean((x2 - enc['x2_rec'])**2)),
        'mse_x2_pred': float(np.mean((x2 - enc['x2_pred'])**2)),
        'ev_x1_rec': explained_variance(x1, enc['x1_rec']),
        'ev_x2_rec': explained_variance(x2, enc['x2_rec']),
        'ev_x2_pred': explained_variance(x2, enc['x2_pred']),
        # area-weighted gridpoint ACC
        'acc_x1_rec': acc_flat(x1, enc['x1_rec'], w1),
        'acc_x2_rec': acc_flat(x2, enc['x2_rec'], w2),
        'acc_x2_pred': acc_flat(x2, enc['x2_pred'], w2),
        # spatial pattern correlation, averaged over seasons
        'pcorr_x2_pred': float(np.nanmean(pattern_corr(x2, enc['x2_pred'], w2))),
        # latent prediction: p(z2 | z1, u) vs encoded z2
        'z2_pred_mse': float(np.mean((enc['z2_mean'] - enc['pz2'])**2)),
        'z2_pred_r_mean': float(np.nanmean(corr_cols(enc['z2_mean'], enc['pz2']))),
    }
    kl1 = kl_per_dim(enc['z1_mean'], enc['z1_log_var'])
    kl2 = kl_per_dim(enc['z2_mean'], enc['z2_log_var'], enc['pz2'])
    m['active_z1'] = int((kl1 > 0.01).sum())     # "active" = mean KL above 0.01 nats
    m['active_z2'] = int((kl2 > 0.01).sum())
    return m, kl1, kl2


def print_seed_report(seed, m_tr, m_te, l1_pen, secs):
    rows = [('total loss', 'loss_total'), ('rec loss x1 (weighted)', 'loss_rec_x1'),
            ('rec loss x2 (weighted)', 'loss_rec_x2'), ('KL z1', 'loss_kl_z1'),
            ('KL z2 | p(z2|z1,u)', 'loss_kl_z2'), ('pred loss x2 (weighted)', 'loss_pred_x2'),
            ('MSE x1 rec', 'mse_x1_rec'), ('MSE x2 rec', 'mse_x2_rec'), ('MSE x2 pred', 'mse_x2_pred'),
            ('expl. var x1 rec', 'ev_x1_rec'), ('expl. var x2 rec', 'ev_x2_rec'),
            ('expl. var x2 pred', 'ev_x2_pred'),
            ('ACC x1 rec', 'acc_x1_rec'), ('ACC x2 rec', 'acc_x2_rec'),
            ('ACC x2 pred (z1->x2)', 'acc_x2_pred'), ('pattern corr x2 pred', 'pcorr_x2_pred'),
            ('latent: r(z2, pz2) mean', 'z2_pred_r_mean'), ('active dims z1', 'active_z1'),
            ('active dims z2', 'active_z2')]
    print(f'\n── seed {seed}  ({secs:.0f} s) ' + '─' * 40)
    print(f'{"":26s}{"train":>10s}{"test":>10s}')
    for label, k in rows:
        print(f'{label:26s}{m_tr[k]:10.3f}{m_te[k]:10.3f}')
    print(f'{"L1 penalty (pred model)":26s}{l1_pen:10.3f}')


def seed_dir(cfg, seed):
    return os.path.join(cfg['out_dir'], f'seed_{seed:03d}')


def run_seed(seed, cfg, data):
    """Train, evaluate, plot and save one seed.

    Returns a dict with: metrics (flat dict), hist (Keras history), monitor (ACC during
    training), enc_tr / enc_te (encode_all outputs), kl1 / kl2 (train KL per dim) and the
    trained models.
    """
    _setup(cfg)
    out = seed_dir(cfg, seed)
    os.makedirs(out, exist_ok=True)
    bs = cfg['batch_size']

    encoder, decoder, pred_model, vae = build_all(seed, cfg, data)
    monitor = SkillMonitor(encoder, decoder, pred_model, data.x1_test, data.x2_test, data.u_test,
                           data.x2.area_weights, every=cfg['acc_every'],
                           print_every=cfg['print_every'])
    print(f'\n▶ training seed {seed}')
    t0 = time.time()
    h = vae.fit([data.x1_train, data.x2_train, data.u_train], None,
                validation_data=([data.x1_test, data.x2_test, data.u_test], None),
                epochs=cfg['epochs'], batch_size=bs, shuffle=True, verbose=0,
                callbacks=[monitor, keras.callbacks.TerminateOnNaN()])
    secs = time.time() - t0
    hist = {k: [float(v) for v in vals] for k, vals in h.history.items()}

    enc_tr = encode_all(encoder, decoder, pred_model, data.x1_train, data.x2_train, data.u_train, bs)
    enc_te = encode_all(encoder, decoder, pred_model, data.x1_test, data.x2_test, data.u_test, bs)
    m_tr, kl1_tr, kl2_tr = split_metrics(vae, enc_tr, data.x1_train, data.x2_train, data.u_train, data, bs)
    m_te, _, _ = split_metrics(vae, enc_te, data.x1_test, data.x2_test, data.u_test, data, bs)
    l1_pen = float(tf.add_n(pred_model.losses)) if pred_model.losses else 0.0
    print_seed_report(seed, m_tr, m_te, l1_pen, secs)

    if cfg.get('make_plots', True):
        # imported here because visualisation imports the metric functions from this module
        import visualisation as vis
        show_for = cfg.get('show_figs_for_seeds', 'all')
        show = show_for == 'all' or seed in (show_for or [])
        vis.plot_seed_figures(seed, out, show, cfg, data, hist, monitor, enc_tr, enc_te,
                              kl1_tr, kl2_tr, decoder, pred_model)

    # save weights, history, latents
    encoder.save_weights(f'{out}/encoder.h5')
    decoder.save_weights(f'{out}/decoder.h5')
    pred_model.save_weights(f'{out}/prediction_model.h5')
    with open(f'{out}/history.json', 'w') as f:
        json.dump(dict(history=hist, acc_epochs=monitor.epochs, acc_pred=monitor.acc_pred,
                       acc_rec=monitor.acc_rec), f)
    np.savez(f'{out}/latents.npz',
             **{f'train_{k}': v for k, v in enc_tr.items()},
             **{f'test_{k}': v for k, v in enc_te.items()},
             years_train=data.years_train, years_test=data.years_test)

    metrics = {'seed': seed, 'train_time_s': secs, 'l1_penalty': l1_pen, 'epochs_run': len(hist['loss'])}
    metrics.update({f'train_{k}': v for k, v in m_tr.items()})
    metrics.update({f'test_{k}': v for k, v in m_te.items()})
    return dict(metrics=metrics, hist=hist,
                monitor=dict(epochs=monitor.epochs, acc_pred=monitor.acc_pred, acc_rec=monitor.acc_rec),
                enc_tr=enc_tr, enc_te=enc_te, kl1=kl1_tr, kl2=kl2_tr,
                models=dict(encoder=encoder, decoder=decoder, prediction_model=pred_model))


def run_all_seeds(cfg, data):
    """run_seed for every seed in cfg['seeds']; returns {seed: result}."""
    return {seed: run_seed(seed, cfg, data) for seed in cfg['seeds']}


def load_seed_models(seed, cfg, data, results=None):
    """(encoder, decoder, prediction_model) of one seed: taken from `results` if they are
    still in memory, otherwise rebuilt and loaded from the weights saved in out_dir.

    Note: rebuilding calls K.clear_session(), so do not mix with live models of other seeds
    from the same session.
    """
    if results is not None and 'models' in results.get(seed, {}):
        m = results[seed]['models']
        return m['encoder'], m['decoder'], m['prediction_model']
    enc, dec, pm, _ = build_all(seed, cfg, data)
    d = seed_dir(cfg, seed)
    enc.load_weights(f'{d}/encoder.h5')
    dec.load_weights(f'{d}/decoder.h5')
    pm.load_weights(f'{d}/prediction_model.h5')
    return enc, dec, pm


# ═════════════════════════════════════════════════════════════════════════════
# Across-seed summary
# ═════════════════════════════════════════════════════════════════════════════
KEY_COLS = ['test_loss_total', 'test_loss_rec_x1', 'test_loss_rec_x2', 'test_loss_kl_z1',
            'test_loss_kl_z2', 'test_loss_pred_x2', 'test_acc_x1_rec', 'test_acc_x2_rec',
            'test_acc_x2_pred', 'test_pcorr_x2_pred', 'test_ev_x2_pred', 'test_z2_pred_r_mean',
            'train_acc_x2_pred', 'train_active_z1', 'train_active_z2']


def metrics_table(results, cfg):
    """Per-seed metrics as a DataFrame (saved to out_dir/metrics_per_seed.csv); prints the
    key columns and their spread across seeds."""
    df = pd.DataFrame([r['metrics'] for r in results.values()]).set_index('seed')
    df.to_csv(os.path.join(cfg['out_dir'], 'metrics_per_seed.csv'))
    with pd.option_context('display.width', 200, 'display.max_columns', 30, 'display.precision', 3):
        print('\nPer-seed metrics (full table in metrics_per_seed.csv):')
        print(df[KEY_COLS])
        print('\nAcross seeds:')
        print(df[KEY_COLS].agg(['mean', 'std', 'min', 'max']))
    return df


def linear_baseline(k, data):
    """Linear benchmark with the same bottleneck: regress x2 on the first k PCs of x1 plus
    u (fitted on train). Returns the test-set prediction of x2."""
    x1_tr, x2_tr = data.x1_train, data.x2_train
    mu1, mu2 = x1_tr.mean(0), x2_tr.mean(0)
    _, _, Vt = np.linalg.svd(x1_tr - mu1, full_matrices=False)
    P = Vt[:k].T

    def design(x1, u):
        return np.c_[(x1 - mu1) @ P, u, np.ones(len(x1))]

    B, *_ = np.linalg.lstsq(design(x1_tr, data.u_train), x2_tr - mu2, rcond=None)
    return design(data.x1_test, data.u_test) @ B + mu2


def baseline_and_ensemble(results, df, cfg, data):
    """Test ACC of the seed-ensemble-mean x2 prediction and of the linear baseline."""
    w2 = data.x2.area_weights
    x2_pred_base = linear_baseline(cfg['latent_dim_z1'], data)
    x2_pred_ens = np.mean([r['enc_te']['x2_pred'] for r in results.values()], axis=0)
    acc_base = acc_flat(data.x2_test, x2_pred_base, w2)
    acc_ens = acc_flat(data.x2_test, x2_pred_ens, w2)
    print(f'\nTest ACC x2 prediction — seeds: {df.test_acc_x2_pred.mean():.3f} ± '
          f'{df.test_acc_x2_pred.std():.3f}'
          f'  |  ensemble mean of {len(results)} seeds: {acc_ens:.3f}'
          f'  |  linear PCA({cfg["latent_dim_z1"]}) regression baseline: {acc_base:.3f}')
    return dict(x2_pred_ens=x2_pred_ens, x2_pred_base=x2_pred_base, acc_ens=acc_ens, acc_base=acc_base)


def sweep_values(x, n, sweep='std', sigma=2.0):
    """n equidistant values spanning the samples x (1-D), in the units of x.

    sweep = 'std'     mean - sigma*std ... mean + sigma*std   (symmetric about the mean)
            'minmax'  min(x) ... max(x)                       (the range seen in the data)
    """
    x = np.asarray(x, dtype=float)
    if sweep == 'std':
        return x.mean() + np.linspace(-sigma, sigma, n) * x.std()
    if sweep == 'minmax':
        return np.linspace(x.min(), x.max(), n)
    raise ValueError(f"sweep must be 'std' or 'minmax', got {sweep!r}")


def _sweep_setting(cfg, sweep):
    return cfg.get('response_sweep', 'std') if sweep is None else sweep


def seed_mean_z1_response(results, cfg, data, ref_seed=None, sigma=2.0, sweep=None):
    """x2 response to each z1 direction, averaged over seeds.

    For every seed, each z1 dim (matched and sign-aligned to `ref_seed` with
    match_latents) is swept in cfg['traversal_n'] equidistant steps, with the other dims
    at their mean and u at its training mean. The sweep is mapped through p(z2|z1,u) and
    decoded to x2, relative to the decoded latent mean of that seed.

    sweep  'std':    mean ± `sigma` std of that seed's z1 dim (training set)
           'minmax': from the min to the max of that seed's z1 dim over the training set
                     (run in the direction that is sign-aligned with ref_seed). Step k is
                     then "the k-th of n points across the observed range" in every seed;
                     its value in std units differs a little between seeds.
           None:     cfg['response_sweep'] (default 'std')

    Returns dict with
        fields     [n_seeds, d1, n_steps, x2 features]   per-seed responses (network space)
        ens        [d1, n_steps, x2 features]             seed mean
        agree      same shape as ens: fraction of seeds with the same sign as the mean
                   (NaN where the mean response is exactly zero)
        match_r    [n_seeds, d1]  |r| of the matched z1 dims with those of ref_seed
        steps      [d1, n_steps]  the sweep of ref_seed in std units from the mean
        steps_all  [n_seeds, d1, n_steps]  the same for every seed (sign-aligned)
        sweep, seeds, ref_seed
    """
    sweep = _sweep_setting(cfg, sweep)
    seeds = list(results)
    ref_seed = seeds[0] if ref_seed is None else ref_seed
    n = cfg['traversal_n']
    z1_ref = results[ref_seed]['enc_tr']['z1_mean']
    d1 = z1_ref.shape[1]
    u_mean = np.tile(data.u_train.mean(0), (n, 1)).astype('float32')

    fields = np.zeros((len(seeds), d1, n, data.x2.n_features))
    steps_all = np.zeros((len(seeds), d1, n))
    match_r = np.zeros((len(seeds), d1))
    for i, s in enumerate(seeds):
        z1 = results[s]['enc_tr']['z1_mean']
        perm, sign, r = match_latents(z1_ref, z1)
        match_r[i] = r
        _, dec, pm = load_seed_models(s, cfg, data, results)

        # this seed's prediction at the latent mean -> responses are anomalies relative to it
        z_mean = z1.mean(0, keepdims=True).astype('float32')
        base = dec.predict([z_mean, pm.predict([z_mean, u_mean[:1]], verbose=0)], verbose=0)[1]
        for k in range(d1):
            j = perm[k]
            vals = sweep_values(z1[:, j], n, sweep, sigma)
            if sign[k] < 0:              # run the sweep in the direction aligned with ref
                vals = vals[::-1]
            steps_all[i, k] = sign[k] * (vals - z1[:, j].mean()) / z1[:, j].std()
            zs = np.tile(z_mean, (n, 1))
            zs[:, j] = vals
            pz2 = pm.predict([zs, u_mean], verbose=0)
            fields[i, k] = dec.predict([zs, pz2], verbose=0)[1] - base

    ens = fields.mean(0)
    agree = (np.sign(fields) == np.sign(ens)[None]).mean(0)
    agree[ens == 0] = np.nan      # e.g. the 0-sigma step: no response, so no stippling
    return dict(fields=fields, ens=ens, agree=agree, match_r=match_r,
                steps=steps_all[seeds.index(ref_seed)], steps_all=steps_all, sweep=sweep,
                seeds=seeds, ref_seed=ref_seed)


def u_sweep(decoder, pred_model, z1_mean, data, n=7, sweep='std', sigma=2.0, steps=None):
    """x2 response to each covariate for one trained model.

    Covariate k is swept in n equidistant steps, the other covariates are held at their
    training mean and z1 at `z1_mean` (one latent vector, usually the training-mean
    posterior mean). Each u is mapped through p(z2|z1,u) and decoded to x2; the decoded x2
    at u = mean is subtracted.

    sweep  'std':    training mean ± `sigma` std
           'minmax': training min ... training max of covariate k
    steps  optional explicit sweep in std units from the mean ([n] or [n_cov, n]);
           overrides n / sweep / sigma.

    u only enters the model through the prior mean of z2 (W z1 + V u + c), so this is
    the x2 pattern the model attributes to u with z1 held fixed. The shift in z2 is linear
    in u (V[k] * du), but the decoder is not, so the response need not be symmetric.

    Returns (fields [n_cov, n_steps, x2 features] in network space,
             steps  [n_cov, n_steps] the sweep in std units from the training mean).
    """
    u_tr = data.u_train
    u_mean, u_std = u_tr.mean(0), u_tr.std(0)
    n_cov = data.n_covariates
    if steps is None:
        steps = np.stack([(sweep_values(u_tr[:, k], n, sweep, sigma) - u_mean[k]) / u_std[k]
                          for k in range(n_cov)])
    steps = np.broadcast_to(np.asarray(steps, dtype=float), (n_cov, np.shape(steps)[-1]))
    n = steps.shape[1]
    z1 = np.tile(np.ravel(z1_mean), (n, 1)).astype('float32')
    u0 = u_mean[None].astype('float32')
    base = decoder.predict([z1[:1], pred_model.predict([z1[:1], u0], verbose=0)], verbose=0)[1]
    out = np.zeros((n_cov, n, data.x2.n_features))
    for k in range(n_cov):
        us = np.tile(u_mean, (n, 1))
        us[:, k] = u_mean[k] + steps[k] * u_std[k]
        pz2 = pred_model.predict([z1, us.astype('float32')], verbose=0)
        out[k] = decoder.predict([z1, pz2], verbose=0)[1] - base
    return out, np.array(steps)


def covariate_response(results, cfg, data, sigma=2.0, sweep=None):
    """x2 response to each covariate u, per seed and averaged over seeds.

    Every covariate is swept in cfg['traversal_n'] equidistant steps (see u_sweep), with
    z1 held at each seed's training-mean posterior mean. Unlike for z1, no matching across
    seeds is needed: u is an input, so it has the same meaning and sign in every seed, and
    the sweep is identical in all seeds.

    sweep  'std':    training mean ± `sigma` std
           'minmax': training min ... training max of each covariate
           None:     cfg['response_sweep'] (default 'std')

    Returns dict with
        fields   [n_seeds, n_cov, n_steps, x2 features]   per-seed responses (network space)
        ens      [n_cov, n_steps, x2 features]             seed mean
        agree    same shape as ens: fraction of seeds with the same sign as the mean
                 (NaN where the mean response is exactly zero)
        steps    [n_cov, n_steps] the sweep in std units from the training mean
        sweep, seeds, names (covariate names)
        u_std    physical std of each covariate (training seasons), to label the steps
        units    units of each covariate ('' if unknown)
    """
    sweep = _sweep_setting(cfg, sweep)
    seeds = list(results)
    n = cfg['traversal_n']
    fields = np.zeros((len(seeds), data.n_covariates, n, data.x2.n_features))
    steps = None
    for i, s in enumerate(seeds):
        _, dec, pm = load_seed_models(s, cfg, data, results)
        fields[i], steps = u_sweep(dec, pm, results[s]['enc_tr']['z1_mean'].mean(0), data,
                                   n=n, sweep=sweep, sigma=sigma)
    ens = fields.mean(0)
    agree = (np.sign(fields) == np.sign(ens)[None]).mean(0)
    agree[ens == 0] = np.nan
    u_std = data.u_raw.isel(season_year=slice(None, data.n_train)).std('season_year').values
    return dict(fields=fields, ens=ens, agree=agree, steps=steps, sweep=sweep, seeds=seeds,
                names=list(data.covariate_names), u_std=u_std,
                units=list(getattr(data, 'covariate_units', [''] * data.n_covariates)))
