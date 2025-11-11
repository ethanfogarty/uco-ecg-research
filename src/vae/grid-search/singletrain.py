import pandas
import tensorflow as tf
from tensorflow import keras
#keras.backend.set_floatx("float64")
keras.config.set_dtype_policy("float32")
from keras import layers
import pyreadr
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


data = pyreadr.read_r("/media/ethan/Data/Documents/UCO/Research/Data Sets/Brno/brno_vae_sets_ones_only.RData")

# ------------------------------------------------------------------
#             Train Data  ------------------------------------------
# ------------------------------------------------------------------
train_data = data['vae.sig.train']
train_data = train_data.to_numpy()

### Max - Norm with train set max value --------------------------------------------
train_data = train_data[..., np.newaxis]
train_data /= 12955.911823647295
#-----------------------------------------------------------------------------------

### Sci-Kit Learn row-wise Z-Score Norm --------------------------------------------
# This method is a nice 1 liner, however the row mean and standard deviation are
#   not saved for later use.
#train_data = StandardScaler().fit_transform(train_data.T).T   # column‑wise z‑score
#train_data = train_data[..., np.newaxis]
#-----------------------------------------------------------------------------------

### Manual row-wise Z-Score Norm ---------------------------------------------------
# This method is configured so that the values for the row mean and standard deviation
#   are saved so that we can project the values back to original scale later
#   on, if needed.
#row_mean = train_data.mean(axis=1, keepdims=True)
#row_stddev = train_data.std(axis=1, keepdims=True)
#train_data = (train_data - row_mean) / row_stddev
#train_data = train_data[..., np.newaxis]
#-----------------------------------------------------------------------------------

train_labels = data['vae.ann.train']
train_labels = train_labels.to_numpy()


# ------------------------------------------------------------------
#             Test Data  -------------------------------------------
# ------------------------------------------------------------------
test_data = data['vae.sig.test']
test_data = test_data.to_numpy()

### Max - Norm with train set max value ---------------------------------------------
test_data = test_data[..., np.newaxis]
test_data /= 12955.911823647295
#------------------------------------------------------------------------------------

### Z - Score Norm ------------------------------------------------------------------
#test_data = StandardScaler().fit_transform(test_data.T).T   # column‑wise z‑score
#test_data = test_data[..., np.newaxis]
#------------------------------------------------------------------------------------

### Manual row-wise Z-Score Norm ---------------------------------------------------
# This method is configured so that the values for the row mean and standard deviation
#   are saved so that we can project the values back to original scale later
#   on, if needed.
#test_data = (test_data - row_mean) / row_stddev
#test_data = test_data[..., np.newaxis]
#-----------------------------------------------------------------------------------

test_labels = data['vae.ann.test']
test_labels = test_labels.to_numpy()



def TrainOne(runID, paramCombo):
    def EncoderCreate(layer_count, unit_count, latent_dim):
        encoder_inputs = keras.Input(shape=(1250, 1))
        x = layers.Conv1D(unit_count[0], 3, activation="relu", padding="same")(encoder_inputs)
        x = layers.MaxPool1D()(x)
        x = layers.Conv1D(unit_count[1], 3, activation="relu", padding="same")(x)
        x = layers.MaxPool1D()(x)
        if (layer_count == 3):
            x = layers.Conv1D(unit_count[2], 3, activation="relu", padding="same")(x)
            x = layers.MaxPool1D()(x)
        x = layers.Flatten()(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name="encoder")
        return encoder

    class Sampler(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            eps = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * eps

    def DecoderCreate(layer_count, unit_count, latent_dim):
        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(units=latent_dim, activation="relu")(latent_inputs)
        x = layers.Reshape(target_shape=(latent_dim, 1))(x)
        if (layer_count == 2):
            x = layers.Conv1DTranspose(unit_count[1], 3, activation="relu", strides=2, padding="same")(x)
            x = layers.Conv1DTranspose(unit_count[0], 3, activation="relu", strides=2, padding="same")(x)
        elif (layer_count == 3):
            x = layers.Conv1DTranspose(unit_count[2], 3, activation="relu", strides=2, padding="same")(x)
            x = layers.Conv1DTranspose(unit_count[1], 3, activation="relu", strides=2, padding="same")(x)
            x = layers.Conv1DTranspose(unit_count[0], 3, activation="relu", padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=1250, activation="linear")(x)
        decoder_outputs = layers.Reshape(target_shape=(1250, 1))(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        return decoder

    # --- VAE Model ---
    class VAE(keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.sampler = Sampler()
            # Track losses
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
            self.kl_weight = 0.0
            self.anneal_rate = 0.01  # This value can be adjusted
            self.max_kl_weight = 1.0

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

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
                    tf.reduce_mean(tf.square(data - reconstruction), axis=[0])
                )
                # KL divergence
                kl_loss = -0.5 * tf.reduce_mean(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                )
                total_loss = reconstruction_loss + self.kl_weight * kl_loss
                #total_loss = reconstruction_loss + kl_loss
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

    # ------------------------------------------------------------------
    #           Create model with 1 combo of grid search params  -------
    # ------------------------------------------------------------------
    latent_dim = paramCombo[0]
    unit_count = paramCombo[1]
    layer_count = paramCombo[2]
    encoder = EncoderCreate(layer_count, unit_count, latent_dim)
    decoder = DecoderCreate(layer_count, unit_count, latent_dim)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(0.0001))
    vae.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=15,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1
    )

    # ------------------------------------------------------------------
    #           Train Model  -------------------------------------------
    # ------------------------------------------------------------------
    history = vae.fit(train_data, epochs=100, batch_size=128, callbacks=[early_stop])

    # ------------------------------------------------------------------
    #           Predict with model and calculate reconstruction MSE  ---
    # ------------------------------------------------------------------
    test_predictions = vae.predict(test_data)
    mse = tf.reduce_mean((test_predictions - test_data) ** 2)
    # Use .numpy() to convert a tf tensor to a numpy array
    print("MSE = ", mse.numpy())
    results = {"param_combo": paramCombo, "mse": mse}

    # ------------------------------------------------------------------
    #           Save test MSE and training graphs to CSV  --------------
    # ------------------------------------------------------------------
    pandas.DataFrame(results).to_csv(
        f"/media/ethan/Data/Documents/UCO/Research/Grid Search Results/Brno_CNNVAE_GS/id_{runID}_results.csv", index=False)
    pandas.DataFrame(history.history).to_csv(
        f"/media/ethan/Data/Documents/UCO/Research/Grid Search Results/Brno_CNNVAE_GS/id_{runID}_traingraph.csv", index=False)


    # ------------------------------------------------------------------
    #           Plot test data vs reconstructions  ---------------------
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(test_predictions[20], label="Test Predictions", color="red")
    plt.plot(test_data[20], label="Test Data", color="blue")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title("Reconstructed vs Original Window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"/media/ethan/Data/Documents/UCO/Research/Grid Search Results/Brno_CNNVAE_GS/id_{runID}_plot.png")