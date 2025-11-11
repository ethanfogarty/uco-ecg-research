import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras
import numpy as np
import pandas as pd
from vgg_model import VGG
import pyreadr


data = pyreadr.read_r("/path/to/dataset.RData")
# ------------------------------------------------------------------
#             Train Data  ------------------------------------------
# ------------------------------------------------------------------
train_data = data['class.signal.train']
train_data = train_data.to_numpy()[..., np.newaxis]

train_labels = data['class.annot.train']
train_labels = train_labels.to_numpy()
train_one_hot_labels = np.eye(3)[train_labels.astype(int) - 1]     # subtract 1 to shift labels to 0-based index
train_one_hot_labels = np.squeeze(train_one_hot_labels, axis=1)
# ------------------------------------------------------------------
#             Test Data  -------------------------------------------
# ------------------------------------------------------------------
test_data = data['class.signal.test']
test_data = test_data.to_numpy()[..., np.newaxis]

test_labels = data['class.annot.test']
test_labels = test_labels.to_numpy()
# ------------------------------------------------------------------
del data    # Clean up data variable to make space in the environment

vgg = VGG()
vgg.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
vgg.summary()

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
history = vgg.fit(train_data, train_one_hot_labels, epochs=100, batch_size=128, callbacks=[early_stop])

# ------------------------------------------------------------------
#           Predict with model and generate confusion matrix -------
# ------------------------------------------------------------------
predictions = vgg.predict(test_data)
# For each predicted window, choose the class with the highest probability
predictions = np.argmax(predictions, axis=1)
cm = tf.math.confusion_matrix(
    # The '-1' in the two lines below is to shift the classes from 1-3 to 0-2
    #   since Python starts indicies from 0. This change affects nothing except
    #   for removing unnecessary '0' columns from the final confusion matrix.
    labels=(test_labels - 1),
    predictions=predictions
).numpy()
# Correcting the class names since the range was shifted above
class_names = [1, 2, 3]
# Construct and print the confusion matrix
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print("\nConfusion Matrix:\n", cm_df)

# ------------------------------------------------------------------
#    Save confusion matrix, params, and training graphs to CSV  ----
# ------------------------------------------------------------------
pd.DataFrame(cm_df).to_csv(
    f"src/results/classifier/vgg_conf_matrix.csv", index=False)
pd.DataFrame(history.history).to_csv(
    f"src/results/classifier/vgg_traingraph.csv", index=False)


