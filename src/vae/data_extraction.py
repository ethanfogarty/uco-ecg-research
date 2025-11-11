import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras
import pyreadr
import numpy as np
import pandas
from Scripts.brno_cnnvae_106.m106 import VAE, Sampler

# Max value from VAE train set 12955.911823647295
data = pyreadr.read_r("/path/to/dataset.RData")
# ------------------------------------------------------------------
#             DS2 Train Data  --------------------------------------
# ------------------------------------------------------------------
train_data = data['class.sig.train']
train_data = train_data.to_numpy()

### Max - Norm with DS1 train set max value ----------------------------------------
train_data = train_data[..., np.newaxis]
train_data /= 12955.911823647295
#-----------------------------------------------------------------------------------

train_labels = data['class.ann.train']
train_labels = train_labels.to_numpy()


# ------------------------------------------------------------------
#             DS2 Test Data  ---------------------------------------
# ------------------------------------------------------------------
test_data = data['class.sig.test']
test_data = test_data.to_numpy()

### Max - Norm with DS1 train set max value -----------------------------------------
test_data = test_data[..., np.newaxis]
test_data /= 12955.911823647295
#------------------------------------------------------------------------------------

test_labels = data['class.ann.test']
test_labels = test_labels.to_numpy()

# ------------------------------------------------------------------
#           Load Saved Model Weights  ------------------------------
# ------------------------------------------------------------------
new_vae = keras.models.load_model("src/vae/weights/brno_cnnvae_106/106.keras", compile=False)

# ------------------------------------------------------------------
#           Extract Encoded DS2 Data  ------------------------------
# ------------------------------------------------------------------
new_z_mean, new_z_logv = new_vae.encoder.predict(train_data, batch_size=128, verbose=1)
ds2_x_train = tf.concat([new_z_mean, new_z_logv], axis=1)
# Convert to float64 since R uses 64 bit double
pandas.DataFrame(ds2_x_train, dtype=np.float64).to_csv("datasets/brno_ds2_x_train_encoded.csv", index = False)

new_z_mean, new_z_logv = new_vae.encoder.predict(test_data, batch_size=128, verbose=1)
ds2_x_test = tf.concat([new_z_mean, new_z_logv], axis=1)
# Convert to float64 since R uses 64 bit double
pandas.DataFrame(ds2_x_test, dtype=np.float64).to_csv("datasets/brno_ds2_x_test_encoded.csv", index = False)

