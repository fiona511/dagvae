#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models.py
=========

Keras building blocks for the three-block conditional VAE.

Structure
---------
    x1 --enc1--> z1 ~ N(0, I)                           --dec1--> x1_hat
    x2 --enc2--> z2 ~ N(pz2(z1), I)          (prior)    --dec2--> x2_hat
    x3 --enc3--> z3 ~ N(pz3(z1, z2), I)      (prior)    --dec3--> x3_hat

Each block has its own encoder and decoder branch. The blocks are coupled
only through the conditional priors, which the prediction model supplies
(linear, optionally with LASSO, so the z1 -> z2 -> z3 links can be read off
its weights).

Sections
--------
    1. Sampling helpers, seeding & regulariser
    2. Encoders
    3. Prediction models (conditional priors)
    4. Decoders
    5. Loss helpers
    6. VAE assembly (build_vae)
    7. Archive

Keras version
-------------
Uses Model.add_loss / Model.add_metric on symbolic tensors. These exist only
in Keras 2 (TensorFlow <= 2.15, or tf_keras). Everything is imported from
tensorflow.keras so Keras 2 and Keras 3 objects are never mixed.

@author: fionaspuler
"""

# =============================================================================
# Imports
# =============================================================================

import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate, Dense, Input, Lambda, Layer
from tensorflow.keras.losses import mse
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import Regularizer, l1


# =============================================================================
# 1. Sampling helpers, seeding & regulariser
# =============================================================================

def sampling(args):
    """
    Reparameterisation trick: z = mean + exp(0.5 * log_var) * eps, eps ~ N(0, I).
    `args` = [z_mean, z_log_var], each of shape (batch, latent_dim).
    """
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# Log-variance of the noise added by sampling_pz.
# The original code used K.exp(0.5), i.e. log_var = 1 (std ~ 1.65). The KL
# terms in build_vae assume the conditional prior has UNIT variance
# (log_var = 0). Kept at 1.0 to reproduce existing results; set to 0.0 to
# make sampling consistent with the KL terms.
PZ_LOG_VAR = 1.0
_PZ_NOISE_STD = math.exp(0.5 * PZ_LOG_VAR)


def sampling_pz(z_mean):
    """Sample from the conditional prior N(z_mean, exp(PZ_LOG_VAR) * I)."""
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]))
    return z_mean + _PZ_NOISE_STD * epsilon


def set_seed(seed, deterministic_ops=False):
    """
    Seed Python, NumPy and TensorFlow. With deterministic_ops=True, also force
    deterministic TF kernels (needed for bit-identical GPU runs; slower).
    """
    keras.utils.set_random_seed(seed)
    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()


def reinitialise_weights(model):
    """
    Redraw every kernel and bias in `model` (recursing into nested models)
    from a fresh copy of that layer's initialiser. Call after set_seed() to
    make the starting weights depend only on the seed. Changes the model IN
    PLACE, so any trained weights are lost.
    """
    for layer in model.layers:
        if isinstance(layer, Model):
            reinitialise_weights(layer)
            continue
        for attr in ('kernel', 'bias'):
            var = getattr(layer, attr, None)
            init = getattr(layer, f'{attr}_initializer', None)
            if var is None or init is None:
                continue
            fresh = init.__class__.from_config(init.get_config())
            var.assign(fresh(var.shape, dtype=var.dtype))


def _is_yes(flag):
    """Accept 'yes'/'no' strings (original API) as well as booleans."""
    if isinstance(flag, str):
        return flag.strip().lower() in ('yes', 'y', 'true', '1')
    return bool(flag)


class CombinedGroupLasso(Regularizer):
    """
    Group-lasso penalty on a Dense kernel of shape (input_dim, output_dim):
      row_strength * sum_i ||W[i, :]||  -> switches off whole INPUT dimensions
      col_strength * sum_j ||W[:, j]||  -> switches off whole OUTPUT dimensions
    Use as Dense(..., kernel_regularizer=CombinedGroupLasso(0.1, 0.1)).
    """

    def __init__(self, row_strength=0.1, col_strength=0.1):
        self.row_strength = row_strength
        self.col_strength = col_strength

    def __call__(self, x):
        # the 1e-8 keeps the gradient finite when a group is exactly zero
        row_norms = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1) + 1e-8)  # (input_dim,)
        col_norms = tf.sqrt(tf.reduce_sum(tf.square(x), axis=0) + 1e-8)  # (output_dim,)
        return (self.row_strength * tf.reduce_sum(row_norms)
                + self.col_strength * tf.reduce_sum(col_norms))

    def get_config(self):
        return {'row_strength': float(self.row_strength),
                'col_strength': float(self.col_strength)}


# =============================================================================
# 2. Encoders
# =============================================================================

def _build_encoder(original_dims, latent_dims, hidden_dims, activation_function):
    """
    One independent MLP branch per input block.
    Outputs: [z1_mean, z1_log_var, z1, z2_mean, z2_log_var, z2, z3_mean, z3_log_var, z3]
    Layers are created in the same order as the original code (all hidden
    layers first, then the heads), so automatic layer names are unchanged.
    """
    inputs = [Input(shape=(d,), name=f'encoder_input_{i}')
              for i, d in enumerate(original_dims, start=1)]

    # hidden layers of each branch
    hidden = []
    for x in inputs:
        h = x
        for units in hidden_dims:
            h = Dense(units, activation=activation_function)(h)
        hidden.append(h)

    # mean / log-variance heads and sampling
    outputs = []
    for i, (h, ld) in enumerate(zip(hidden, latent_dims), start=1):
        z_mean = Dense(ld, name=f'z{i}_mean')(h)
        z_log_var = Dense(ld, name=f'z{i}_log_var')(h)
        z = Lambda(sampling, output_shape=(ld,), name=f'z{i}')([z_mean, z_log_var])
        outputs += [z_mean, z_log_var, z]

    return Model(inputs, outputs, name='encoder')


def build_encoder_2L(original_dim_z1, original_dim_z2, original_dim_z3,
                     latent_dim_z1, latent_dim_z2, latent_dim_z3,
                     dim_layer1, dim_layer2, activation_function):
    """Encoder with two hidden layers per block: input -> dim_layer1 -> dim_layer2 -> z."""
    return _build_encoder([original_dim_z1, original_dim_z2, original_dim_z3],
                          [latent_dim_z1, latent_dim_z2, latent_dim_z3],
                          [dim_layer1, dim_layer2], activation_function)


def build_encoder_3L(original_dim_z1, original_dim_z2, original_dim_z3,
                     latent_dim_z1, latent_dim_z2, latent_dim_z3,
                     dim_layer1, dim_layer2, activation_function, dim_layer3=32):
    """Encoder with three hidden layers per block (third layer was hard-coded to 32)."""
    return _build_encoder([original_dim_z1, original_dim_z2, original_dim_z3],
                          [latent_dim_z1, latent_dim_z2, latent_dim_z3],
                          [dim_layer1, dim_layer2, dim_layer3], activation_function)


# =============================================================================
# 3. Prediction models (conditional priors)
# =============================================================================
# All prediction models take [z1, z2] and return
#     [pz2_z1, pz3_z1z2, pz3_sampled]
# pz2_z1   : prior mean of z2 given z1
# pz3_z1z2 : prior mean of z3 given (z1, z2)
# pz3_sampled : a sample from N(pz3_z1z2, exp(PZ_LOG_VAR) I)

def build_prediction_model_linear(latent_dim_z1, latent_dim_z2, latent_dim_z3,
                                  regularization='yes', regularization_term=0.01):
    """
    Linear conditional priors. With regularization='yes' (or True) both maps
    get an L1 (LASSO) penalty of strength `regularization_term`.
    """
    reg = l1(regularization_term) if _is_yes(regularization) else None

    z1_input = Input(shape=(latent_dim_z1,), name='z1_input')
    z2_input = Input(shape=(latent_dim_z2,), name='z2_input')
    z1z2 = Concatenate()([z1_input, z2_input])

    pz2_z1 = Dense(latent_dim_z2, name='pz2_z1', kernel_regularizer=reg)(z1_input)
    pz3_z1z2 = Dense(latent_dim_z3, name='pz3_z1z2', kernel_regularizer=reg)(z1z2)
    pz3_sampled = Lambda(sampling_pz, output_shape=(latent_dim_z3,),
                         name='pz3_sampled')(pz3_z1z2)

    return Model([z1_input, z2_input], [pz2_z1, pz3_z1z2, pz3_sampled],
                 name='prediction_model')


# =============================================================================
# 4. Decoders
# =============================================================================

def _build_decoder(latent_dims, hidden_dims, activation_function, original_dims):
    """One independent MLP branch per latent block; linear output layer."""
    latent_inputs = [Input(shape=(ld,), name=f'z{i}_sampling')
                     for i, ld in enumerate(latent_dims, start=1)]

    outputs = []
    for i, (z, od) in enumerate(zip(latent_inputs, original_dims), start=1):
        h = z
        for units in hidden_dims:
            h = Dense(units, activation=activation_function)(h)
        outputs.append(Dense(od, name=f'outputs_z{i}')(h))

    return Model(latent_inputs, outputs, name='decoder')


def build_decoder_2L(latent_dim_z1, latent_dim_z2, latent_dim_z3,
                     dim_layer1, dim_layer2, activation_function,
                     original_dim_z1, original_dim_z2, original_dim_z3):
    """Decoder mirroring build_encoder_2L: z -> dim_layer2 -> dim_layer1 -> output."""
    return _build_decoder([latent_dim_z1, latent_dim_z2, latent_dim_z3],
                          [dim_layer2, dim_layer1], activation_function,
                          [original_dim_z1, original_dim_z2, original_dim_z3])


def build_decoder(latent_dim_z1, latent_dim_z2, latent_dim_z3,
                  dim_layer1, dim_layer2, activation_function,
                  original_dim_z1, original_dim_z2, original_dim_z3, dim_layer3=32):
    """Decoder mirroring build_encoder_3L: z -> dim_layer3 -> dim_layer2 -> dim_layer1 -> output."""
    return _build_decoder([latent_dim_z1, latent_dim_z2, latent_dim_z3],
                          [dim_layer3, dim_layer2, dim_layer1], activation_function,
                          [original_dim_z1, original_dim_z2, original_dim_z3])


# clearer alias
build_decoder_3L = build_decoder


# =============================================================================
# 5. Loss helpers
# =============================================================================

def _kl_to_gaussian(mean, log_var, prior_mean=0.0):
    """KL( N(mean, exp(log_var)) || N(prior_mean, I) ), summed over latent dims."""
    return -0.5 * tf.reduce_sum(
        1 + log_var - tf.square(mean - prior_mean) - tf.exp(log_var), axis=-1)


def _offdiag_cov_penalty(z):
    """Sum of squared off-diagonal entries of the batch covariance of z."""
    z_centered = z - tf.reduce_mean(z, axis=0)
    n = tf.cast(tf.shape(z)[0], tf.float32)
    cov = tf.matmul(z_centered, z_centered, transpose_a=True) / n
    off_diag = cov - tf.linalg.diag(tf.linalg.diag_part(cov))
    return tf.reduce_sum(tf.square(off_diag))


def _add_loss_and_metric(model, value, name):
    """Add the batch mean of `value` as a loss term and log it under `name`."""
    value = K.mean(value)
    model.add_loss(value)
    model.add_metric(value, name=name, aggregation='mean')


def _add_reconstruction_losses(model, inputs, outputs, original_dims, factors):
    """
    Per-block reconstruction loss = sum of squared errors * factor
    (mse * dim == SSE). Logged as reconstruction_loss_z1..3.
    """
    for i, (x, x_hat, dim, f) in enumerate(zip(inputs, outputs, original_dims, factors),
                                           start=1):
        _add_loss_and_metric(model, mse(x, x_hat) * dim * f, f'reconstruction_loss_z{i}')


def _prepare_seed(seed, deterministic_ops, submodels):
    """Seed everything, then redraw the sub-models' weights so they depend on the seed."""
    if seed is None:
        return
    set_seed(seed, deterministic_ops)
    for m in submodels:              # fixed order -> reproducible
        reinitialise_weights(m)


class ScheduledWeight(Layer):
    """
    Multiplies its input by a scalar, non-trainable weight `scale` that a
    callback can change during training (see WarmupSchedule).
    """

    def __init__(self, initial_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_value = float(initial_value)

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(), trainable=False,
                                     initializer=keras.initializers.Constant(self.initial_value))
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        return {**super().get_config(), 'initial_value': self.initial_value}


def _add_loss_and_metric(model, value, name, weight_layer=None):
    """
    Add the batch mean of `value` as a loss term and log it under `name`.
    With a `weight_layer` (ScheduledWeight), the LOSS is multiplied by the
    scheduled weight, but the METRIC stays unweighted, so logged values are
    comparable across epochs and runs.
    """
    value = K.mean(value)
    model.add_metric(value, name=name, aggregation='mean')
    model.add_loss(weight_layer(value) if weight_layer is not None else value)


# =============================================================================
# 6. VAE assembly
# =============================================================================

def build_vae(encoder, decoder, prediction_model,
              original_dim_z1, original_dim_z2, original_dim_z3,
              reconstruction_loss_factors,
              latent_dim_z1, latent_dim_z2, latent_dim_z3,
              include_orthogonality_constraint=True, chosen_learning_rate=0.001,
              orthogonality_loss_factor=5, seed=0, deterministic_ops=False,
              warmup=False):
    """
    Assemble and compile the joint VAE.

    Loss = sum_k reconstruction_k                 (SSE * reconstruction_loss_factors[k])
         + w_z1   * KL(q(z1|x1) || N(0, I))
         + w_cond * KL(q(z2|x2) || N(pz2(z1), I))
         + w_cond * KL(q(z3|x3) || N(pz3(z1, z2), I))
         + w_orth * orthogonality_loss_factor * off-diagonal covariance of z3_mean  (optional)
         + L1 penalties inside the prediction model

    seed
        Seeds Python/NumPy/TF and then RE-INITIALISES the weights of encoder,
        prediction_model and decoder, so the starting weights depend only on
        `seed`. The models are changed in place. Pass seed=None to keep the
        current weights (e.g. when rebuilding around trained sub-models).
    deterministic_ops
        Also force deterministic TF kernels (bit-identical GPU runs; slower).
    warmup
        False: all weights w are 1 (identical to the previous version).
        True:  w_z1, w_cond, w_orth are ScheduledWeight layers named
               'kl_weight_z1', 'kl_weight_cond', 'ortho_weight'. They start
               at 0 (reconstruction-only training) and are ramped up by a
               WarmupSchedule callback passed to fit() (see default_warmup).
               Logged regularization_loss_* / orthogonality_loss metrics are
               UNWEIGHTED; 'loss' / 'val_loss' are the weighted totals.
               The schedule weights are saved with the model, so weight files
               from warmup=True only load into models built with warmup=True.

    latent_dim_z1..3 are unused but kept for backward compatibility.
    """
    _prepare_seed(seed, deterministic_ops, [encoder, prediction_model, decoder])

    inputs = [Input(shape=(d,), name=f'encoder_input_{i}')
              for i, d in enumerate([original_dim_z1, original_dim_z2, original_dim_z3],
                                    start=1)]

    # encoder
    (z1_mean, z1_log_var, z1,
     z2_mean, z2_log_var, z2,
     z3_mean, z3_log_var, z3) = encoder(inputs)

    # conditional prior means (from the deterministic latent means)
    pz2_z1, pz3_z1z2, _ = prediction_model([z1_mean, z2_mean])

    # decoder
    outputs = decoder([z1, z2, z3])

    vae = Model(inputs, outputs, name='vae_joint')

    # scheduled loss weights (None = constant 1)
    if warmup:
        w_kl_z1 = ScheduledWeight(0.0, name='kl_weight_z1')
        w_kl_cond = ScheduledWeight(0.0, name='kl_weight_cond')   # shared by z2 and z3
        w_ortho = ScheduledWeight(0.0, name='ortho_weight')
    else:
        w_kl_z1 = w_kl_cond = w_ortho = None

    # reconstruction (never scheduled)
    _add_reconstruction_losses(vae, inputs, outputs,
                               [original_dim_z1, original_dim_z2, original_dim_z3],
                               reconstruction_loss_factors)

    # KL / regularisation
    _add_loss_and_metric(vae, _kl_to_gaussian(z1_mean, z1_log_var),
                         'regularization_loss_z1', w_kl_z1)
    _add_loss_and_metric(vae, _kl_to_gaussian(z2_mean, z2_log_var, pz2_z1),
                         'regularization_loss_z2', w_kl_cond)
    _add_loss_and_metric(vae, _kl_to_gaussian(z3_mean, z3_log_var, pz3_z1z2),
                         'regularization_loss_z3', w_kl_cond)

    # decorrelate the z3 dimensions
    if include_orthogonality_constraint:
        _add_loss_and_metric(vae,
                             _offdiag_cov_penalty(z3_mean) * orthogonality_loss_factor,
                             'orthogonality_loss', w_ortho)

    vae.compile(optimizer=Adam(learning_rate=chosen_learning_rate))
    return vae

# =============================================================================
# 7. Training schedule (warm-up)
# =============================================================================

class WarmupSchedule(keras.callbacks.Callback):
    """
    Linear warm-up of the ScheduledWeight layers of a model built with
    build_vae(..., warmup=True).

    schedules : {layer_name: (start_epoch, ramp_epochs, final_value)}
        The weight is 0 before start_epoch, then rises linearly to final_value
        over ramp_epochs, and stays there.

    The current values are also written to the epoch logs, so they appear in
    history.history (e.g. history.history['kl_weight_cond']).
    """

    def __init__(self, schedules):
        super().__init__()
        self.schedules = dict(schedules)

    @staticmethod
    def value_at(epoch, start_epoch, ramp_epochs, final_value):
        if epoch < start_epoch:
            return 0.0
        if ramp_epochs <= 0:
            return float(final_value)
        return float(final_value) * min(1.0, (epoch - start_epoch + 1) / ramp_epochs)

    def on_epoch_begin(self, epoch, logs=None):
        for name, schedule in self.schedules.items():
            self.model.get_layer(name).scale.assign(self.value_at(epoch, *schedule))

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for name in self.schedules:
                logs[name] = float(self.model.get_layer(name).scale.numpy())


def default_warmup(recon_epochs=10, ramp_epochs=30, cond_delay=0):
    """
    Reconstruction only for `recon_epochs`, then the KL and orthogonality
    terms ramp linearly to full weight over `ramp_epochs`.
    cond_delay: extra epochs before the conditional KLs (z2, z3) start, so
    the prediction model gets a settled target. Everything is at full weight
    from epoch recon_epochs + cond_delay + ramp_epochs.
    """
    return WarmupSchedule({
        'kl_weight_z1':   (recon_epochs,              ramp_epochs, 1.0),
        'kl_weight_cond': (recon_epochs + cond_delay, ramp_epochs, 1.0),
        'ortho_weight':   (recon_epochs,              ramp_epochs, 1.0),
    })