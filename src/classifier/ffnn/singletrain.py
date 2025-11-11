import pandas as pd
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras
from keras import layers
import pyreadr
import numpy as np


data = pyreadr.read_r("/path/to/dataset.RData")
# ------------------------------------------------------------------
#             Train Data  ------------------------------------------
# ------------------------------------------------------------------
train_data = pd.read_csv("/path/to/dataset.csv")
train_labels = data['class.annot.train']
train_labels.columns = ['label'] # Rename the column from '0' to 'label'
train_one_hot_labels = np.eye(3)[train_labels.astype(int) - 1]     # subtract 1 to shift labels to 0-based index
train_one_hot_labels = np.squeeze(train_one_hot_labels, axis=1)
# ------------------------------------------------------------------
#             Test Data  -------------------------------------------
# ------------------------------------------------------------------
test_data = pd.read_csv("/path/to/dataset.csv")
test_labels = data['class.annot.test']
test_labels.columns = ['label'] # Rename the column from '0' to 'label'
# --------------------------------------------------------------------

def TrainOne(runID, paramCombo):
    # --- FFNN Classifier ---
    def FFNN_Create(layer_count, unit_count):
        if (layer_count == 3):
            model = keras.Sequential([
                layers.Dense(unit_count[0], activation='relu', input_shape=(400,)),
                layers.Dense(unit_count[1], activation='relu'),
                layers.Dense(unit_count[2], activation='relu'),
                layers.Dense(3, activation='softmax')
            ])
        elif (layer_count == 4):
            model = keras.Sequential([
                layers.Dense(unit_count[0], activation='relu', input_shape=(400,)),
                layers.Dense(unit_count[1], activation='relu'),
                layers.Dense(unit_count[2], activation='relu'),
                layers.Dense(unit_count[3], activation='relu'),
                layers.Dense(3, activation='softmax')
            ])
        elif (layer_count == 5):
            model = keras.Sequential([
                layers.Dense(unit_count[0], activation='relu', input_shape=(400,)),
                layers.Dense(unit_count[1], activation='relu'),
                layers.Dense(unit_count[2], activation='relu'),
                layers.Dense(unit_count[3], activation='relu'),
                layers.Dense(unit_count[4], activation='relu'),
                layers.Dense(3, activation='softmax')
            ])
        elif (layer_count == 10):
            model = keras.Sequential([
                layers.Dense(unit_count[0], activation='relu', input_shape=(400,)),
                layers.Dense(unit_count[1], activation='relu'),
                layers.Dense(unit_count[2], activation='relu'),
                layers.Dense(unit_count[3], activation='relu'),
                layers.Dense(unit_count[4], activation='relu'),
                layers.Dense(unit_count[5], activation='relu'),
                layers.Dense(unit_count[6], activation='relu'),
                layers.Dense(unit_count[7], activation='relu'),
                layers.Dense(unit_count[8], activation='relu'),
                layers.Dense(unit_count[9], activation='relu'),
                layers.Dense(3, activation='softmax')
            ])
        return model


    # ------------------------------------------------------------------
    #           Create model with 1 combo of grid search params  -------
    # ------------------------------------------------------------------
    layer_count = paramCombo[0]
    unit_count = paramCombo[1]
    ffnn = FFNN_Create(layer_count, unit_count)
    ffnn.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    ffnn.summary()

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
    history = ffnn.fit(train_data.to_numpy(), train_one_hot_labels, epochs=100, batch_size=256, callbacks=[early_stop])

    # ------------------------------------------------------------------
    #           Predict with model and generate confusion matrix -------
    # ------------------------------------------------------------------
    predictions = ffnn.predict(test_data.to_numpy())
    # For each predicted window, choose the class with the highest probability
    predictions = np.argmax(predictions, axis=1)
    cm = tf.math.confusion_matrix(
        # The '-1' in the two lines below is to shift the classes from 1-3 to 0-2
        #   since Python starts indicies from 0. This change affects nothing except
        #   for removing unnecessary '0' columns from the final confusion matrix.
        labels=(test_labels.to_numpy() - 1),
        predictions=predictions
    )
    # Recall per class = TP / (TP + FN)
    sensitivity_per_class = tf.linalg.diag_part(cm) / tf.reduce_sum(cm, axis=1)
    # Balanced accuracy
    balanced_accuracy = tf.reduce_mean(sensitivity_per_class)
    print("\nMulti-Class Balanced Accuracy: ", balanced_accuracy)
    # Correcting the class names since the range was shifted above
    class_names = [1, 2, 3]
    # Construct and print the confusion matrix
    cm_df = pd.DataFrame(cm.numpy(), index=class_names, columns=class_names)
    print("\nConfusion Matrix:\n", cm_df)

    # ------------------------------------------------------------------
    #    Save confusion matrix, params, and training graphs to CSV  ----
    # ------------------------------------------------------------------
    pd.DataFrame(cm_df).to_csv(
        f"src/results/classifier/ffnn/id_{runID}_cm.csv", index=False)
    pd.DataFrame(paramCombo).to_csv(
        f"src/results/classifier/ffnn/id_{runID}_params.csv", index=False)
    pd.DataFrame(history.history).to_csv(
        f"src/results/classifier/ffnn/id_{runID}_traingraph.csv", index=False)
