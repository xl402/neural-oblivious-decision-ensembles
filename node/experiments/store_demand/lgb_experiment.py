import numpy as np
import time

import lightgbm as lgb
import pandas as pd

from node.experiments.store_demand.features import load_data
from node.experiments.store_demand.features import add_features
from node.experiments.store_demand.features import add_fold_label


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    # taking exponential since we were predicting log(sales+1)
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


def lgb_validation(params, lgbtrain, lgbval, X_val, Y_val, verbose_eval):
    t0 = time.time()
    evals_result = {}
    model = lgb.train(params, lgbtrain, num_boost_round=params['num_boost_round'],
                      valid_sets=[lgbtrain, lgbval], feval=lgbm_smape,
                      early_stopping_rounds=params['early_stopping_rounds'],
                      evals_result=evals_result, verbose_eval=verbose_eval)
    print(model.best_iteration)
    print('Total time taken to build the model: ', (time.time()-t0)/60, 'minutes!!')
    pred_Y_val = model.predict(X_val, num_iteration=model.best_iteration)
    pred_Y_val = np.expm1(pred_Y_val)
    Y_val = np.expm1(Y_val)
    val_df = pd.DataFrame(columns=['true_Y_val','pred_Y_val'])
    val_df['pred_Y_val'] = pred_Y_val
    val_df['true_Y_val'] = Y_val
    print(val_df.shape)
    print(val_df.sample(5))
    print('SMAPE for validation data is:{}'.format(smape(pred_Y_val, Y_val)))
    return model, val_df


def get_feature_engineered_data():
    df = load_data()
    df = add_fold_label(df)
    df = add_features(df)
    return df


def get_train_val_data(df):
    x_train = df.loc[df.fold == 'train', :]
    x_val = df.loc[df.fold == 'val', :]

    y_train = x_train.sales.values.reshape((-1))
    y_val = x_val.sales.values.reshape((-1))
    return x_train, x_val, y_train, y_val


if __name__ == '__main__':
    df = get_feature_engineered_data()
    x_train, x_val, y_train, y_val = get_train_val_data(df)

    non_feature_cols = ['date', 'sales', 'fold', 'id', 'year']
    cols = [col for col in x_train.columns if col not in non_feature_cols]

    lgbtrain = lgb.Dataset(data=x_train.loc[:, cols].values, label=y_train,
                           feature_name=cols)
    lgbval = lgb.Dataset(data=x_val.loc[:, cols].values, label=y_val,
                         reference=lgbtrain, feature_name=cols)

    lgb_params = {'task': 'train',
                  'boosting_type': 'gbdt',
                  'objective': 'regression',
                  'metric': {'mae'},
                  'num_leaves': 10,
                  'learning_rate': 0.02,
                  'feature_fraction': 0.8,
                  'max_depth': 5,
                  'num_boost_round': 15000,
                  'early_stopping_rounds': 200,
                  'verbose': 0,
                  'nthread': -1}

    model, val_df = lgb_validation(lgb_params,
                                   lgbtrain,
                                   lgbval,
                                   x_val.loc[:, cols].values,
                                   y_val,
                                   verbose_eval=500)
