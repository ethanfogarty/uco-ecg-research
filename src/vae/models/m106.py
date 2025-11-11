import tensorflow as tf
from tensorflow import keras
from keras import layers

# Latent dimension
latent_dim = 200


# --- Encoder ---
encoder_inputs = keras.Input(shape=(1250,1))
x = layers.Conv1D(32, 3, activation="relu", padding="same")(encoder_inputs)
x = layers.MaxPool1D()(x)
x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPool1D()(x)
x = layers.Flatten()(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
encoder.summary()

@keras.utils.register_keras_serializable()
class Sampler(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


# --- Decoder ---
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(units=latent_dim, activation="relu")(latent_inputs)
x = layers.Reshape(target_shape=(latent_dim, 1))(x)
x = layers.Conv1DTranspose(64, 3, activation="relu", strides = 2, padding="same")(x)
x = layers.Conv1DTranspose(32, 3, activation="relu", strides = 2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(units=1250, activation="linear")(x)
decoder_outputs = layers.Reshape(target_shape=(1250, 1))(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


# --- VAE Model ---
#@keras.utils.register_keras_serializable(package="VAE106", name="VAE")
@keras.utils.register_keras_serializable()
class VAE(keras.Model):
    def __init__(self, encoder, decoder, sampler, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.kl_weight = 0.0
        self.anneal_rate = 0.01
        self.max_kl_weight = 1.0

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def get_config(self):
        base_config = super().get_config()
        custom_config = {
            "encoder": self.encoder,
            "decoder": self.decoder,
            "sampler": self.sampler
        }
        return {
            **base_config,
            **custom_config
        }

    @classmethod
    def from_config(cls, config):
        encoder_cfg = config.pop("encoder")
        decoder_cfg = config.pop("decoder")
        sampler_cfg = config.pop("sampler")
        encoder = keras.utils.deserialize_keras_object(encoder_cfg)
        decoder = keras.utils.deserialize_keras_object(decoder_cfg)
        sampler = keras.utils.deserialize_keras_object(sampler_cfg)

        return cls(encoder=encoder, decoder=decoder, sampler=sampler, **config)

    def call(self, inputs):
        """Defines the forward pass for validation and inference."""
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampler([z_mean, z_log_var])
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler([z_mean, z_log_var])
            reconstruction = self.decoder(z)
            # Reconstruction loss
            reconstruction_loss = tf.reduce_sum(
                tf.reduce_mean(tf.square(data - reconstruction), axis=[0])  # sum over pixels/features
            )
            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            total_loss = reconstruction_loss + self.kl_weight * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        # KL weight annealing
        self.kl_weight = min(self.max_kl_weight, self.kl_weight + self.anneal_rate)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

