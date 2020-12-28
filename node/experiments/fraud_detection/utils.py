import numpy as np
import pandas as pd
import tensorflow as tf
import math
import os
from sklearn.metrics import roc_auc_score
from node.experiments.fraud_detection.config import Config


# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
# WARNING! THIS CAN DAMAGE THE DATA
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def load_data(file):
    path = os.path.join(Config.data_dir, file)
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
        args={}):

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
