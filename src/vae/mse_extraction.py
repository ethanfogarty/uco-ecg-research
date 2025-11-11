import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras
import pyreadr
import numpy as np
import pandas
from Scripts.brno_cnnvae_106.m106 import VAE, Sampler


data = pyreadr.read_r("/path/to/dataset.RData")
# ------------------------------------------------------------------
#             DS2 Train Data  --------------------------------------
# ------------------------------------------------------------------
train_data = data['class.signal.train'].to_numpy()
#train_data = train_data.to_numpy()
### Max - Norm with DS1 train set max value ----------------------------------------
train_data = train_data[..., np.newaxis]
train_data /= 12955.911823647295
#-----------------------------------------------------------------------------------
train_labels = data['class.annot.train'].to_numpy()
#train_labels = train_labels.to_numpy()


# ------------------------------------------------------------------
#             DS2 Test Data  ---------------------------------------
# ------------------------------------------------------------------
test_data = data['class.signal.test'].to_numpy()
#test_data = test_data.to_numpy()
### Max - Norm with DS1 train set max value -----------------------------------------
test_data = test_data[..., np.newaxis]
test_data /= 12955.911823647295
#------------------------------------------------------------------------------------
test_labels = data['class.annot.test'].to_numpy()
#test_labels = test_labels.to_numpy()


# ------------------------------------------------------------------
#           Load Saved Model Weights  ------------------------------
# ------------------------------------------------------------------
new_vae = keras.models.load_model("src/vae/weights/brno_cnnvae_106/106.keras", compile=False)


# ------------------------------------------------------------------
#           Calculate MSE of reconstructed train set ---------------
# ------------------------------------------------------------------
reconstruction = new_vae.predict(train_data, batch_size=128, verbose=1)
train_ds2_mse = tf.reduce_mean(np.square(reconstruction - train_data), axis=[1, 2])
# Convert to float64 since R uses 64 bit double
pandas.DataFrame(train_ds2_mse.numpy()).to_csv("datasets/brno_ds2_train_mse.csv", index = False)


# ------------------------------------------------------------------
#           Calculate MSE of reconstructed test set ----------------
# ------------------------------------------------------------------
reconstruction = new_vae.predict(test_data, batch_size=128, verbose=1)
test_ds2_mse = tf.reduce_mean(np.square(reconstruction - test_data), axis=[1, 2])
# Convert to float64 since R uses 64 bit double
pandas.DataFrame(test_ds2_mse.numpy()).to_csv("datasets/brno_ds2_test_mse.csv", index = False)
