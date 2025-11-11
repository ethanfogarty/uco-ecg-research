import tensorflow as tf
from tensorflow import keras
import pyreadr
import numpy as np
import matplotlib.pyplot as plt
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from m106 import VAE, encoder, decoder, Sampler

data = pyreadr.read_r("/path/to/dataset.RData")

# ------------------------------------------------------------------
#             Train Data  ------------------------------------------
# ------------------------------------------------------------------
train_data = data['vae.sig.train']
train_data = train_data.to_numpy()

### Max - Norm with train set max value --------------------------------------------
train_data = train_data[..., np.newaxis]
train_data /= 12955.911823647295
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

test_labels = data['vae.ann.test']
test_labels = test_labels.to_numpy()



vae = VAE(encoder, decoder, Sampler())
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
vae.fit(train_data, epochs=100, batch_size=128, callbacks=[early_stop])

# ------------------------------------------------------------------
#           Predict with model and calculate reconstruction MSE  ---
# ------------------------------------------------------------------
test_predictions = vae.predict(test_data)
mse = tf.reduce_mean((test_predictions - test_data) ** 2)
# Use .numpy() to convert a tf tensor to a numpy array
print("MSE = ", mse.numpy())

keras.models.save_model(vae, "src/vae/weights/brno_cnnvae_106/106.keras")


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
plt.savefig(f"src/results/vae/106_plot.png")
