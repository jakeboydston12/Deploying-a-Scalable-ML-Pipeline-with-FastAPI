import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
# TODO: load the cencus.csv data

data_path = os.path.join(os.path.dirname(__file__), "../data/census.csv")
data = pd.read_csv(data_path)

# TODO: split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# TODO: use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# TODO: use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

# save the model and the encoder
save_model(model, "model/model.pkl")
save_model(encoder, "model/encoder.pkl")
save_model(lb, "model/lb.pkl")

# load the model
model = load_model(
    model_path
) 

# TODO: use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Overall Metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {fbeta:.3f}")

# TODO: compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features
with open("slice_output.txt", "w") as f:
    for col in cat_features:
        f.write(f"\n--- Performance Slices for Feature: {col} ---\n")
        for value in test[col].unique():
            precision, recall, fbeta = performance_on_categorical_slice(
                test, col, value, cat_features, "salary", encoder, lb, model
            )
            output = f"{value}: Precision {precision:.3f}, Recall {recall:.3f}, F1 {fbeta:.3f}"
            f.write(output + "\n")

print("Training complete. Model and slice performance saved.")
