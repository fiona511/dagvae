"""DAG-VAE model: two latent spaces linked by a learned prior.

x1 and x2 each have their own encoder and decoder branch (no weight sharing). The only
connection between them is the prior of z2: instead of N(0, I) it is a unit-variance
Gaussian whose mean is a linear function of z1 (posterior mean) and the covariates u. An L1
penalty on W encourages a sparse z1 -> z2 graph.

Loss (all terms averaged over the batch):

    a1 * D1 * MSE(x1, x̂1)                          reconstruction x1   'reconstruction_loss_z1'
  + a2 * D2 * MSE(x2, x̂2)                          reconstruction x2   'reconstruction_loss_z2'
  + c  * D2 * MSE(x2, dec_x2(W z1 + V u + c))      prediction of x2    'prediction_loss_x2'
  + b1 * KL( q(z1|x1) || N(0, I) )                                     'regularization_loss_z1'
  + b2 * KL( q(z2|x2) || N(W z1 + V u + c, I) )                        'regularization_loss_z2'
  + L1 penalties on W (and optionally V)

with D1, D2 the number of input features (so MSE * D = summed squared error),
a = reconstruction_loss_factors, b = kl_loss_factors, c = prediction_loss_factor.
The metric names in quotes are what Keras reports in the training history; the "_z1/_z2"
suffix refers to the x1/z1 and x2/z2 branch respectively (names kept from the original code
so old history files stay comparable).

Written for the Keras 2 API (tf.keras in TensorFlow <= 2.15, or tf_keras with
TF_USE_LEGACY_KERAS=1): it relies on Model.add_loss / Model.add_metric.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam


def sampling(args):
    """Reparameterisation trick: z = mean + exp(log_var / 2) * eps, eps ~ N(0, I)."""
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def _encoder_branch(x, hidden_dims, activation, dropout_rate):
    """Dense stack, widest layer first; dropout after the first (widest) layer."""
    for i, h in enumerate(hidden_dims):
        x = Dense(h, activation=activation)(x)
        if i == 0:
            x = Dropout(dropout_rate)(x)
    return x


def _decoder_branch(z, hidden_dims, activation, dropout_rate):
    """Mirror of the encoder: narrowest layer first; dropout before the last (widest) layer."""
    layers = list(hidden_dims)[::-1]
    for i, h in enumerate(layers):
        if i == len(layers) - 1 and i > 0:
            z = Dropout(dropout_rate)(z)
        z = Dense(h, activation=activation)(z)
    return z


def build_encoder(input_dim_x1, input_dim_x2, latent_dim_z1, latent_dim_z2,
                  hidden_dims=(128, 64, 32), activation='relu', dropout_rate=0.3):
    
    """Two independent encoders, x1 -> q(z1|x1) and x2 -> q(z2|x2).

    input_dim_x1/x2   number of input features (active grid cells) of x1 / x2
    latent_dim_z1/z2  latent dimensions
    hidden_dims       widths of the hidden layers, input side first, e.g. (128, 64, 32)
    activation        activation of the hidden layers
    dropout_rate      dropout after the first hidden layer

    Returns Model([x1, x2] -> [z1_mean, z1_log_var, z1, z2_mean, z2_log_var, z2]), where
    z1 / z2 are samples drawn with the reparameterisation trick.
    """
    inputs_x1 = Input(shape=(input_dim_x1,), name='encoder_input_1')
    inputs_x2 = Input(shape=(input_dim_x2,), name='encoder_input_2')

    # the x1 branch is built completely before the x2 branch, which (with a fixed seed)
    # keeps the weight initialisation identical to the original code
    h1 = _encoder_branch(inputs_x1, hidden_dims, activation, dropout_rate)
    h2 = _encoder_branch(inputs_x2, hidden_dims, activation, dropout_rate)

    z1_mean = Dense(latent_dim_z1, name='z1_mean')(h1)
    z1_log_var = Dense(latent_dim_z1, name='z1_log_var')(h1)
    z1 = Lambda(sampling, output_shape=(latent_dim_z1,), name='z1')([z1_mean, z1_log_var])

    z2_mean = Dense(latent_dim_z2, name='z2_mean')(h2)
    z2_log_var = Dense(latent_dim_z2, name='z2_log_var')(h2)
    z2 = Lambda(sampling, output_shape=(latent_dim_z2,), name='z2')([z2_mean, z2_log_var])

    return Model([inputs_x1, inputs_x2],
                 [z1_mean, z1_log_var, z1, z2_mean, z2_log_var, z2], name='encoder')


def build_decoder(latent_dim_z1, latent_dim_z2, output_dim_x1, output_dim_x2,
                  hidden_dims=(128, 64, 32), activation='relu', dropout_rate=0.3):
    """Two independent decoders, z1 -> x̂1 and z2 -> x̂2 (linear output layer).

    hidden_dims is given in the same order as for the encoder (input side first); the
    decoder uses it reversed. Dropout is applied before the last (widest) hidden layer.

    Returns Model([z1, z2] -> [x̂1, x̂2]).
    """
    latent_inputs_z1 = Input(shape=(latent_dim_z1,), name='z1_sampling')
    latent_inputs_z2 = Input(shape=(latent_dim_z2,), name='z2_sampling')

    h1 = _decoder_branch(latent_inputs_z1, hidden_dims, activation, dropout_rate)
    outputs_x1 = Dense(output_dim_x1, name='outputs_z1')(h1)

    h2 = _decoder_branch(latent_inputs_z2, hidden_dims, activation, dropout_rate)
    outputs_x2 = Dense(output_dim_x2, name='outputs_z2')(h2)

    return Model([latent_inputs_z1, latent_inputs_z2], [outputs_x1, outputs_x2], name='decoder')


def build_prediction_model_linear(latent_dim_z1, latent_dim_z2, n_covariates=1,
                                  regularization_term=0.01, covariate_reg=0.0):
    """Linear mean of the z2 prior: mu = W z1 + V u + c.

    regularization_term  L1 penalty on W (z1 -> z2 weights, layer 'pz2_z1')
    covariate_reg        L1 penalty on V (u -> z2 weights, layer 'pz2_u'); 0 = none
    The bias c sits in the 'pz2_u' layer.

    Returns Model([z1, u] -> mu).
    """
    z1_input = Input(shape=(latent_dim_z1,), name='z1_input')
    u_input = Input(shape=(n_covariates,), name='u_input')

    from_z1 = Dense(latent_dim_z2, use_bias=False, name='pz2_z1',
                    kernel_regularizer=l1(regularization_term))(z1_input)
    from_u = Dense(latent_dim_z2, use_bias=True, name='pz2_u',
                   kernel_regularizer=l1(covariate_reg) if covariate_reg else None)(u_input)
    pz2_mean = Add(name='pz2_mean')([from_z1, from_u])

    return Model([z1_input, u_input], [pz2_mean], name='prediction_model')


def build_vae(encoder, decoder, prediction_model, input_dim_x1, input_dim_x2,
              reconstruction_loss_factors=(1.0, 1.0), kl_loss_factors=(1.0, 1.0),
              n_covariates=1, chosen_learning_rate=0.001, seed=0,
              prediction_loss_factor=0.0, optimizer=None):
    
    """Connect encoder, decoder and prediction model into the joint, compiled DAG-VAE.

    Inputs [x1, x2, u]; outputs [x̂1, x̂2]. All loss terms are attached with add_loss (see
    module docstring), so train with vae.fit([x1, x2, u], None, ...).

    reconstruction_loss_factors  (a1, a2) weights of the reconstruction terms
    kl_loss_factors              (b1, b2) weights of the KL terms
    prediction_loss_factor       c: weight of MSE(x2, decode(z1, p(z2|z1,u))); 0 = off.
                                 Trains the z1 -> z2 mapping (and z1 and the x2 decoder)
                                 directly on x2 prediction skill.
    seed                         re-seeds Python/NumPy/TF here, i.e. it controls dropout and
                                 latent sampling during training (weight initialisation is
                                 controlled by the seed set before the sub-models are built)
    optimizer                    a Keras optimizer; default Adam(chosen_learning_rate)
    """
    keras.utils.set_random_seed(seed)

    inputs_x1 = Input(shape=(input_dim_x1,), name='encoder_input_1')
    inputs_x2 = Input(shape=(input_dim_x2,), name='encoder_input_2')
    inputs_u = Input(shape=(n_covariates,), name='covariate_input')

    z1_mean, z1_log_var, z1, z2_mean, z2_log_var, z2 = encoder([inputs_x1, inputs_x2])

    # prior mean of z2, from the (deterministic) posterior mean of z1 and the covariates
    pz2_mean = prediction_model([z1_mean, inputs_u])

    outputs = decoder([z1, z2])
    vae = Model([inputs_x1, inputs_x2, inputs_u], outputs, name='vae_joint')

    def _add(term, name):
        vae.add_loss(K.mean(term))
        vae.add_metric(K.mean(term), name=name, aggregation='mean')

    # ── reconstruction: summed squared error per sample (MSE * number of features) ──
    rec_x1 = tf.keras.losses.mse(inputs_x1, outputs[0]) * input_dim_x1 * reconstruction_loss_factors[0]
    rec_x2 = tf.keras.losses.mse(inputs_x2, outputs[1]) * input_dim_x2 * reconstruction_loss_factors[1]
    _add(rec_x1, 'reconstruction_loss_z1')
    _add(rec_x2, 'reconstruction_loss_z2')

    # ── prediction: decode x2 from the prior mean p(z2 | z1, u) and compare with x2 ──
    _, x2_from_prior = decoder([z1_mean, pz2_mean])
    pred_x2 = tf.keras.losses.mse(inputs_x2, x2_from_prior) * input_dim_x2 * prediction_loss_factor
    _add(pred_x2, 'prediction_loss_x2')

    # ── KL terms (closed form for diagonal Gaussians with unit prior variance) ──
    kl_z1 = -0.5 * tf.reduce_sum(
        1 + z1_log_var - tf.math.square(z1_mean) - tf.math.exp(z1_log_var), axis=-1)
    kl_z2 = -0.5 * tf.reduce_sum(
        1 + z2_log_var - tf.math.square(z2_mean - pz2_mean) - tf.math.exp(z2_log_var), axis=-1)
    _add(kl_z1 * kl_loss_factors[0], 'regularization_loss_z1')
    _add(kl_z2 * kl_loss_factors[1], 'regularization_loss_z2')

    vae.compile(optimizer=optimizer or Adam(learning_rate=chosen_learning_rate))
    return vae
