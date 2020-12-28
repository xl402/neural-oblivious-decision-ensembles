import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from node.experiments.fraud_detection.config import Config
from node.experiments.fraud_detection.utils import reduce_mem_usage


def read_data(file):
    df = pd.read_csv(os.path.join(Config.data_dir, file), index_col='TransactionID')
    return df


def add_time_encoding(df):
    # Day of week in which a transaction happened.
    df['Transaction_day_of_week'] = np.floor((df['TransactionDT'] / (3600 * 24) - 1) % 7)
    # Hour of the day in which a transaction happened.
    df['Transaction_hour'] = np.floor(df['TransactionDT'] / 3600) % 24
    return df


def train_test_split(df, test_ratio=0.3):
    split_index = int(len(df) * (1-test_ratio))
    df_train = df.iloc[:split_index]
    df_test = df.iloc[split_index:]
    print(f"--- Train size {len(df_train)} | Test size {len(df_test)} ---")
    return df_train, df_test


def train_test_split_and_save(df, data_dir, tag):
    train, test = train_test_split(df, test_ratio=0.3)
    train.to_csv(os.path.join(data_dir, f'train_{tag}.csv'), index=False)
    test.to_csv(os.path.join(data_dir, f'test_{tag}.csv'), index=False)


if __name__ == '__main__':

    # Read in data
    transaction = read_data('train_transaction.csv')
    identity = read_data('train_identity.csv')

    # Merge data
    data = transaction.merge(identity, how='left', left_index=True, right_index=True)

    print(f"--- DataFrame shape (before dropping): {data.shape} ---")

    # Get useless columns/features to drop
    # https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm
    cols_to_drop = [col for col in data.columns if col not in Config.useful_features]
    cols_to_drop.remove('isFraud')

    # add time encoding then drop unnecessary columns
    data = add_time_encoding(data)
    data = data.drop(cols_to_drop, axis=1)

    print(f"--- DataFrame shape (after dropping): {data.shape} ---")

    # Label Encoding
    for f in data.columns:
        if data[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))

    # Fill NaNs
    data = data.fillna(-1)
    data = reduce_mem_usage(data)

    # Create a separate dataframe without engineered features beginning with 'V'
    engineered_features_to_drop = [ft for ft in data.columns.tolist() if ft[0] in ['V', 'D', 'C']]
    engineered_features_to_drop.remove('DeviceInfo')
    engineered_features_to_drop.remove('DeviceType')

    data_without_fe = data.drop(engineered_features_to_drop, axis=1)

    # perform train test splits and save - tag indicates the % of "V" features retained
    train_test_split_and_save(data, Config.data_dir, "features")
    print("--- Saved processed dataframe with feature engineering ---")

    train_test_split_and_save(data_without_fe, Config.data_dir, "base")
    print("--- Saved processed dataframe without feature engineering ---")
