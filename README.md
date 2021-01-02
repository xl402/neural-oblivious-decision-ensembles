# NODE Experiments
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/xl402/neural-oblivious-decision-ensembles/node)

Things in research tend to be too good to be true, the original paper does not
compare `NODE` on feature-engineered data (which is essential for making best
performing models). In this branch, we compare NODE against modern tree-based
models such as *lightgbm*, *xgboost* and another DL based tabular architecture
*TabNet*. The aim is to validate the performance of NODE to be true to authors'
claims.


### Initial Setup
Install requirements by running:
```
pip install -r requirements.txt
```
Then export project and data directories to python path:
```
export PYTHONPATH=$PATH_TO_REPO/node
export NODE_DATA_DIR=$PATH_TO_REPO/node/experiments/data
```
There is around 1GB of data to be downloaded.

## 2. Experiments

### 2.1 Fraud Detection
Note that this script imputes all null values with `-1` for DL-based models which may account for some of the weaker performances recorded.
Use `xgboost_experiment.py`, `mlp_experiment.py`, `node_experiment.py`, `tabnet_experiment.py` respectively to train and evaluate models on either dataset (uncommenting the code appropriately to choose between `base` and `features` datasets).

The following tables summarise the performance metrics obtained for each model.

<table>
<tr><th>Raw Data </th><th>With Feature Engineering</th></tr>
<tr><td>

| Model   | Test AUC |
|---------|----------|
| Xgboost           | 0.840|
| NODE (1, 100, 6)  | 0.803|
| MLP (50, 50)      | 0.735|
| TabNet (32, 8, 1) | 0.760|

</td><td>

| Model   | Test AUC |
|---------|----------|
| Xgboost            | 0.898|
| NODE (4, 10, 6)    | 0.836|
| MLP (100, 100)     | 0.783|
| TabNet (128, 8, 1) | 0.834|


</td></tr> </table>

* NODE (n_layers, n_trees, depth)
* MLP (dense_dim_1, dense_dim_2,..., dense_dim_n)
* TabNet (feature_dim, output_dim, num_decision_steps)

**Conclusion**:

Tree based models still set a very strong baseline for classifying fraud, both with only base features and engineered features.
The expectation was that the neural nets would perform (relatively) better with base features only, by learning their own features. Further measures could have been taken to improve neural net performance e.g. downsampling genuines and handling missing values more appropriately. NODE seems to do better than TabNet on base features only, suggesting it is better at generating complex features, but with engineered features, TabNet performs just as well. Both NODE and TabNet significantly outperform simple MLPs (with minimal tuning), but underperform XGBoost using the same basic preprocessing. 

### 2.2 Store Demand Forecasting
We use the dataset from Kaggle's [Store Item Demand Forecasting Challenge
](https://www.kaggle.com/c/demand-forecasting-kernels-only/data)
with feature engineering primarily based on [this public notebook](https://www.kaggle.com/abhilashawasthi/feature-engineering-lgb-model)

It is useless to experiment with raw without feature-engineering data since due
to the complete lack of raw feature columns (i.e. target column sales is one of
the three columns). All parameters are hand-tuned no more than 10 times.

| Model   | Test SMAPE |
|---------|----------|
| Xgboost            | 12.55|
| NODE (5, 20, 5)    | 12.64|
| MLP (100, 100)     | 15.57|
| TabNet (81, 20, 1) | 13.37|

Again, we see NODE comes closest to Lightgbm but does not beat it
out-of-the-box.
