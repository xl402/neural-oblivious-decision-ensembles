import os
import logging

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

DATA_DIR = f"{os.getenv('NODE_DATA_DIR')}/store_demand"
TRAIN_CUT_OFF = '2016-09-01'
VAL_CUT_OFF = '2017-06-01'


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), parse_dates=['date'])
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), parse_dates=['date'])
    df = pd.concat([train, test], sort=False)
    df.drop('id', axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.sort_values(by=['store','item','date'], axis=0, inplace=True)
    logger.info(f"data loaded of shape {df.shape}")
    return df


def add_fold_label(df):
    df['fold'] = 'nan'
    train_mask = (df.date < TRAIN_CUT_OFF)
    val_mask = (df.date >= TRAIN_CUT_OFF) & (df.date < VAL_CUT_OFF)
    test_mask = (df.date >= VAL_CUT_OFF)
    df.loc[train_mask, 'fold'] = 'train'
    df.loc[val_mask, 'fold'] = 'val'
    df.loc[test_mask, 'fold'] = 'test'
    return df


def add_features(df):
    df['sales'] = np.log1p(df.sales.values)
    df = create_time_features(df)
    lags = [91, 98, 105, 112, 119, 126, 182, 364, 546, 728]
    df = create_sales_lag_feats(df, gpby_cols=['store', 'item'], target_col='sales',
                               lags=lags)

    df = create_sales_rmean_feats(df, gpby_cols=['store', 'item'],
                                  target_col='sales', windows=[364, 546],
                                  min_periods=10, win_type='triang')

    df = create_sales_ewm_feats(df, gpby_cols=['store', 'item'],
                                target_col='sales',
                                alpha=[0.95, 0.9, 0.8, 0.7, 0.6, 0.5],
                                shift=lags)
    logger.info(f"features generated of shape {df.shape}")
    return df


def create_time_features(df):
    df['dayofmonth'] = df.date.dt.day
    df['dayofyear'] = df.date.dt.dayofyear
    df['dayofweek'] = df.date.dt.dayofweek
    df['month'] = df.date.dt.month
    df['year'] = df.date.dt.year
    df['weekofyear'] = df.date.dt.weekofyear
    df['is_month_start'] = (df.date.dt.is_month_start).astype(int)
    df['is_month_end'] = (df.date.dt.is_month_end).astype(int)
    return df


def create_sales_lag_feats(df, gpby_cols, target_col, lags):
    gpby = df.groupby(gpby_cols)
    for i in lags:
        values = gpby[target_col].shift(i)
        df['_'.join([target_col, 'lag', str(i)])] = values
    return df


def create_sales_rmean_feats(df, gpby_cols, target_col, windows, min_periods=2,
                             shift=1, win_type=None):
    gpby = df.groupby(gpby_cols)
    for w in windows:
        value = gpby[target_col].shift(shift).rolling(window=w,
                                                      min_periods=min_periods,
                                                      win_type=win_type).mean()
        df['_'.join([target_col, 'rmean', str(w)])] = value
    return df


def create_sales_ewm_feats(df, gpby_cols, target_col, alpha=[0.9], shift=[1]):
    gpby = df.groupby(gpby_cols)
    for a in alpha:
        for s in shift:
            value = gpby[target_col].shift(s).ewm(alpha=a).mean().values
            df['_'.join([target_col, 'lag', str(s), 'ewm', str(a)])] = value
    return df
