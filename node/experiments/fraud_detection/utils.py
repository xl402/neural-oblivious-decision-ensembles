import numpy as np
import pandas as pd
import tensorflow as tf
import math
import os
from sklearn.metrics import roc_auc_score


DATA_DIR = f"{os.getenv('NODE_DATA_DIR')}/fraud_detection"


def load_data(file):
    path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(path)
    y = df['isFraud'].to_numpy()
    x = df.drop('isFraud', axis=1).to_numpy()
    feature_names = df.drop('isFraud', axis=1).columns
    return x, y, feature_names


def create_validation_split(x_train, y_train, val_ratio):
    split_index = math.ceil(len(x_train) * (1-val_ratio))
    x_val, y_val = x_train[split_index:], y_train[split_index:]
    x_train, y_train = x_train[:split_index], y_train[:split_index]
    return x_train, y_train, x_val, y_val


def evaluate_model(x_train, y_train, x_test, y_test, model):
    y_train_predict = model.predict(x_train, batch_size=128)
    y_test_predict = model.predict(x_test, batch_size=128)
    train_auc = roc_auc_score(y_train, y_train_predict)
    test_auc = roc_auc_score(y_test, y_test_predict)
    return train_auc, test_auc


def run_keras_experiment(
        train_data,
        test_data,
        build_model,
        val_ratio=0.2,
        epochs=100,
        batch_size=128,
        patience=5,
        args=None):

    if args is None:
        args = {}

    x_train, y_train, feature_names = load_data(train_data)
    x_test, y_test, _ = load_data(test_data)

    # create validation set
    x_train, y_train, x_val, y_val = create_validation_split(x_train, y_train, val_ratio)
    val_set = (x_val, y_val)

    # Build model
    model = build_model(**args)
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val_set,
        callbacks=tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=patience,
            verbose=1,
            mode='max',
            restore_best_weights=True
        ),
    )

    # evaluate model
    train_auc, test_auc = evaluate_model(x_train, y_train, x_test, y_test, model)
    print(f"Train AUC: {train_auc} | Test AUC {test_auc}")

    return train_auc, test_auc
