# YDF is a faster and more up-to-date version of tfdf
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import pyreadr

data = pyreadr.read_r("/media/ethan/Data/Documents/UCO/Research/Data Sets/Brno/Brno_classifier_sets.RData")

train_data = pd.read_csv("/media/ethan/Data/Documents/UCO/Research/Data Sets/brno_ds2_x_train_encoded.csv")
train_labels = data['class.annot.train']
train_labels.columns = ['label'] # Rename the column from '0' to 'label'

test_data = pd.read_csv("/media/ethan/Data/Documents/UCO/Research/Data Sets/brno_ds2_x_test_encoded.csv")
test_labels = data['class.annot.test']
test_labels.columns = ['label'] # Rename the column from '0' to 'label'

# Uncomment statements below to check for NaN values ------------------------------
#mask_missing = test_labels['label'].isna()
# 1) Row numbers with missing labels
#print("Rows with missing labels:", mask_missing[mask_missing].index.tolist())
# 2) Full rows
#missing_rows = train_data.loc[mask_missing]
#print(missing_rows)
# ----------------------------------------------------------------------------------

# Mark "data" for garbage collection to save memory in environment
del data

rf_model = tfdf.keras.RandomForestModel(
    task = tfdf.keras.Task.CLASSIFICATION,
    random_seed = 3
)

# Train
rf_model.fit(x = train_data.to_numpy(), y = train_labels.to_numpy())

# Evaluate
predictions = rf_model.predict(test_data.to_numpy())
# For each predicted window, choose the class with the highest probability
predictions = np.argmax(predictions, axis=1)
cm = tf.math.confusion_matrix(
    # The '-1' in the two lines below is to shift the classes from 1-3 to 0-2
    #   since Python starts indicies from 0. This change affects nothing except
    #   for removing unnecessary '0' columns from the final confusion matrix.
    labels = (test_labels.to_numpy() - 1),
    predictions = (predictions - 1)
).numpy()
# Correcting the class names since the range was shifted above
class_names = [1, 2, 3]
# Construct and print the confusion matrix
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print("\nConfusion Matrix:\n", cm_df)