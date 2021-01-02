from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


from node.experiments.store_demand.features import add_features
from node.experiments.store_demand.features import add_fold_label
from node.experiments.store_demand.features import load_data
from tabnet import TabNet


def get_train_val_data(df):
    x_train = df.loc[df.fold == 'train', :]
    x_val = df.loc[df.fold == 'val', :]

    y_train = x_train.sales.values.reshape((-1))
    y_val = x_val.sales.values.reshape((-1))
    return x_train, x_val, y_train, y_val


def get_feature_engineered_data():
    df = load_data()
    df = add_fold_label(df)
    df = add_features(df)
    return df


def smape(y_true, y_pred):
    # taking exponential since we were predicting log(sales+1)
    y_true = tf.math.expm1(y_true)
    y_pred = tf.math.expm1(y_pred)
    n = len(y_true)
    masked_arr = ~((y_pred == 0) & (y_true == 0))
    y_pred, y_true = y_pred[masked_arr], y_true[masked_arr]
    num = K.abs(y_pred - y_true)
    denom = K.abs(y_pred) + K.abs(y_true)
    smape = (200 * K.sum(num / denom)) / n
    return smape


if __name__ == '__main__':
    df = get_feature_engineered_data()
    x_train, x_val, y_train, y_val = get_train_val_data(df)

    non_feature_cols = ['date', 'sales', 'fold', 'id', 'year']
    cols = [col for col in x_train.columns if col not in non_feature_cols]

    x_train = x_train.loc[:, cols]
    x_val = x_val.loc[:, cols]
    x_train.fillna(0, inplace=True)
    x_val.fillna(0, inplace=True)

    model = tf.keras.Sequential(
        [
            TabNet(
                feature_columns=None,
                num_features=81,
                feature_dim=20,
                output_dim=8,
                num_decision_steps=1,
                relaxation_factor=1.5,
                sparsity_coefficient=1e-5,
            ),
            Dense(1, activation="relu")
        ]
    )

    es_callback = EarlyStopping(monitor='val_loss',
                                mode='min',
                                patience=10,
                                restore_best_weights=True)

    model.compile(optimizer='adam',
                  loss='mae',
                  metrics=[smape])

    history = model.fit(x_train,
                        y_train,
                        batch_size=100,
                        epochs=500,
                        validation_data=(x_val, y_val),
                        callbacks=[es_callback])

    np.save('results/tabnet_y_val_pred', np.expm1(model.predict(x_val)[:, 0]))
