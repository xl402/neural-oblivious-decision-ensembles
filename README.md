# NODE Experiments
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/xl402/neural-oblivious-decision-ensembles/node)

Things in research tend to be too good to be true, the original paper does not
compare `NODE` on feature-engineered data (which is essential for making best
performing models). In this branch, we compare NODE against modern tree-based
models such as *lightgbm*, *xgboost* and another DL based tabular architecture
*TabNet*. The aim is to validate the performance of NODE to be true to authors'
claims.


## 1. Initial Setup
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
Note that this script very naively imputes all null values with `-1` (which is set as the default `null` value for the Xgboost algorithm), and this may account for some of the weaker performance with the neural network models.

Then use `xgboost_experiment.py`, `mlp_experiment.py`, `node_experiment.py`, `tabnet_experiment.py` respectively to train and evaluate models on either dataset (uncommenting the code appropriately to choose between `base` and `features` datasets).

The following tables summarise the performance metrics obtained for each model.

<table>
<tr><th>Raw Data </th><th>With Feature Engineering</th></tr>
<tr><td>

| Model   | Train AUC | Test AUC |
|---------|-----------|----------|
| Xgboost           | 0.934| 0.840|
| NODE (1, 100, 6)  | 0.839| 0.803|
| NODE (4, 10, 6)   | 0.830| 0.803|
| MLP (50, 50)      | 0.726| 0.735|
| MLP (100, 100)    | 0.718| 0.745|
| TabNet (32, 8, 1) | 0.772| 0.760|
| TabNet (64, 8, 1) | 0.796| 0.767|
| TabNet (128, 8, 1)| 0.796| 0.764|

</td><td>

| Model   | Train AUC | Test AUC |
|---------|-----------|----------|
| Xgboost            | 0.962| 0.898|
| NODE (1, 100, 6)   | 0.893| 0.843|
| NODE (4, 10, 6)    | 0.886| 0.836|
| MLP (50, 50)       | 0.777| 0.796|
| MLP (100, 100)     | 0.775| 0.783|
| TabNet (32, 8, 1)  | 0.878| 0.837|
| TabNet (64, 8, 1)  | 0.861| 0.840|
| TabNet (128, 8, 1) | 0.842| 0.834|


</td></tr> </table>

* NODE (n_layers, n_trees, depth)
* MLP (dense_dim_1, dense_dim_2,..., dense_dim_n)
* TabNet (feature_dim, output_dim, num_decision_steps)

**Conclusion**:


Tree based models still set a very strong baseline for classifying fraud, both with only base features and engineered features. The expectation was that the neural nets would perform (relatively) better with base features only, by learning their own features. Further measures could have been taken to improve neural net performance e.g. downsampling genuines and handling missing values more appropriately. NODE seems to do better than TabNet on base features only, suggesting it is better at generating complex features, but with engineered features, TabNet performs just as well. Both NODE and TabNet significantly outperform simple MLPs (with minimal tuning), but underperform XGBoost using the same basic preprocessing. 
