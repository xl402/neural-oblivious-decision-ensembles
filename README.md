# NODE Experiments
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/xl402/neural-oblivious-decision-ensembles/node)

The original paper does not
compare `NODE` on feature-engineered data (which is essential for making best
performing models). In this branch, we compare NODE against modern tree-based
models such as *LightGBM*, *Xgboost* and another DL based tabular architecture
*TabNet*.


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
We use the dataset from Kaggle's [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)
with feature engineering primarily based on [this public notebook](https://www.kaggle.com/abhilashawasthi/feature-engineering-lgb-model).

The following tables summarise the performance metrics (AUC, higher the better) obtained for each model.

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


### 2.2 Store Demand Forecasting
We use the dataset from Kaggle's [Store Item Demand Forecasting Challenge
](https://www.kaggle.com/c/demand-forecasting-kernels-only/data)
with feature engineering primarily based on [this public notebook](https://www.kaggle.com/abhilashawasthi/feature-engineering-lgb-model).

It is useless to experiment with raw without feature-engineering data since due
to the complete lack of raw feature columns (i.e. target column sales is one of
the three columns). All parameters are hand-tuned no more than 10 times.

Table below summarizes the result (with SMAPE as a metric, lower the better).

| Model   | Test SMAPE |
|---------|----------|
| LightGBM           | 12.55|
| NODE (3, 10, 4)    | 12.57|
| MLP (100, 100)     | 12.62|
| TabNet (81, 20, 1) | 13.37|


## 3. Conclusions
Tree based models remains very competitive. In both experiments, classic tree-based models out-perform DL based models.
NODE seems to do slightly better than TabNet, both significantly outperform MLP
as expected.
