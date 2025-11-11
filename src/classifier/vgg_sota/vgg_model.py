import tensorflow as tf
from tensorflow import keras
from keras import layers

def VGG():
    model = tf.keras.Sequential([
        layers.Conv1D(64, 3, activation="relu", padding="same", input_shape=(1250,1)),
        layers.Conv1D(64, 3, activation="relu", padding="same"),
        layers.MaxPool1D(),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.MaxPool1D(),
        layers.Conv1D(256, 3, activation="relu", padding="same"),
        layers.Conv1D(256, 3, activation="relu", padding="same"),
        layers.Conv1D(256, 3, activation="relu", padding="same"),
        layers.MaxPool1D(),
        layers.Conv1D(512, 3, activation="relu", padding="same"),
        layers.Conv1D(512, 3, activation="relu", padding="same"),
        layers.Conv1D(512, 3, activation="relu", padding="same"),
        layers.MaxPool1D(),
        layers.Conv1D(512, 3, activation="relu", padding="same"),
        layers.Conv1D(512, 3, activation="relu", padding="same"),
        layers.Conv1D(512, 3, activation="relu", padding="same"),
        layers.MaxPool1D(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(4096, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    return model