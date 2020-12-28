import xgboost as xgb
from sklearn.metrics import roc_auc_score
import os
import numpy as np
from table_logger import TableLogger
from node.experiments.fraud_detection.utils import load_data, create_validation_split
from node.experiments.fraud_detection.config import Config


def evaluate_xgboost(x_train, y_train, x_test, y_test, model):

    y_train_predict = model.predict_proba(x_train)[:, 1]
    y_test_predict = model.predict_proba(x_test)[:, 1]

    train_auc = roc_auc_score(y_train, y_train_predict)
    test_auc = roc_auc_score(y_test, y_test_predict)
    return train_auc, test_auc


def save_feature_importance(model, feature_names, tag):

    if not os.path.exists(Config.results_dir):
        os.makedirs(Config.results_dir)

    feature_importance = model.feature_importances_
    sort_index = np.argsort(-feature_importance)

    sorted_importance = feature_importance[sort_index]
    sorted_names = feature_names[sort_index]

    with open(os.path.join(Config.results_dir, f'feature_importance_{tag}.csv'), 'wb') as csvfile:
        tbl = TableLogger(file=csvfile, columns='Feature,Importance')
        for name, importance in zip(sorted_names, sorted_importance):
            tbl(name, importance)


def run_experiment(train_data, test_data, tag, seed=42, save_importance=False):

    x_train, y_train, feature_names = load_data(train_data)
    x_test, y_test, _ = load_data(test_data)

    # create validation set
    x_train, y_train, x_val, y_val = create_validation_split(x_train, y_train, val_ratio=0.2)
    eval_set = [(x_val, y_val)]

    # fit model
    model = xgb.XGBClassifier(n_estimators=2048,
                              n_jobs=8,
                              max_depth=9,
                              learning_rate=0.05,
                              subsample=0.9,
                              colsample_bytree=0.9,
                              use_label_encoder=False,
                              verbosity=1,
                              missing=-1,
                              random_state=seed)
    model.fit(x_train, y_train, eval_metric="auc", eval_set=eval_set, early_stopping_rounds=5, verbose=False)

    # evaluate model
    train_auc, test_auc = evaluate_xgboost(x_train, y_train, x_test, y_test, model)
    print(f"Train AUC: {train_auc} | Test AUC {test_auc}")

    if save_importance:
        save_feature_importance(model, feature_names, tag)

    return train_auc, test_auc


def run_repeated_experiment(train_data, test_data, tag, n_repeats):

    train_auc_list = []
    test_auc_list = []

    if not os.path.exists(Config.results_dir):
        os.makedirs(Config.results_dir)

    with open(os.path.join(Config.results_dir, f'xgboost_repeats_{tag}.csv'), 'wb') as csvfile:
        tbl = TableLogger(file=csvfile, columns='Seed,Train Auc,Test Auc')
        for i in range(n_repeats):
            train_auc, test_auc = run_experiment(train_data, test_data, tag, seed=i)
            train_auc_list.append(train_auc)
            test_auc_list.append(test_auc)
            tbl(i, train_auc, test_auc)
        tbl('Mean', np.mean(train_auc_list), np.mean(test_auc_list))


if __name__ == "__main__":
    train_file, test_file = 'train_base.csv', 'test_base.csv'
    # train_data, test_data = 'train_features.csv', 'test_features.csv'
    run_experiment(train_file, test_file, tag='base_default', save_importance=True)
