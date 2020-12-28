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
Then export project to python path:
```
export PYTHONPATH=$PATH_TO_REPO/node
export NODE_DATA_DIR=$PATH_TO_REPO/node/experiments/data
```
There is around 1GB of data to be downloaded.

## 2. Experiments

### 3. Fraud Detection
First run `preprocessing.py`, which creates the following files:
* `train_base.csv` and `test_base.csv` - datasets with base features only
* `train_features.csv` and `test_features.csv`. 

Note that this script very naively imputes all null values with `-1` (which is set as the default `null` value for the Xgboost algorithm), and this may account for some of the weaker performance with the neural network models.

Then use `xgboost_experiment.py`, `mlp_experiment.py`, `node_experiment.py`, `tabnet_experiment.py` respectively to train and evaluate models on either dataset (uncommenting the code appropriately to choose between `base` and `features` datasets).

The following table summarises the performance metrics obtained for some hand selected hyperparameters for each model.

| Model   | Feature Set | Train AUC | Test AUC |
|---------|-------------|-----------|----------|
| Xgboost | Base        | 0.9341750201668205 | 0.840333479818724 |
| Xgboost | Engineered  | 0.9626651725621306 | 0.898024337734299 |
| NODE (1, 100, 6)    | Base        | 0.8391224436855323 | 0.803909962911713 |
| NODE (1, 100, 6)   | Engineered  | 0.8932972685549408 | 0.843956972193509 |
| NODE (1, 10, 6) | Base   | 0.8289127595871792 | 0.7934694655518505 |
| NODE (2, 10, 6) | Base   | 0.8507937446693866 | 0.8014519164190872 |
| NODE (4, 10, 6) | Base   | 0.8309770927434055 | 0.8034763367229408 |
| NODE (4, 10, 6) | Engineered | 0.8867308472064607 | 0.8367876400535365 |
| MLP (50, 50) | Base   | 0.7268181087095296 | 0.735218857168153  |
| MLP (50, 50) | Engineered   | 0.7779503318444431 | 0.7966996772270184 |
| MLP (100, 100) | Base   | 0.7182236613189015 | 0.7455641818997751 |
| MLP (100, 100) | Engineered   | 0.7752253749421438 | 0.7830629930532992 |
| TabNet (32, 8, 1) | Base | 0.7727588925502363 | 0.7605743161194505 |
| TabNet (32, 8, 1) | Engineered | 0.878419329384196 | 0.8373619552184257 |
| TabNet (64, 8, 1) | Base | 0.796961537428581 | 0.7678338065769991 |
| TabNet (64, 8, 1) | Engineered | 0.8616169521689243 | 0.8407751362546809 |
| TabNet (128, 8, 1) | Base | 0.7963570764755292 | 0.7646962675339493 |
| TabNet (128, 8, 1) | Engineered | 0.8421969164377779 | 0.8346405963240082 |

* NODE (n_layers, n_trees, depth)
* MLP (dense_dim_1, dense_dim_2,..., dense_dim_n)
* TabNet (feature_dim, output_dim, num_decision_steps)

**Conclusion**:


Tree based models still set a very strong baseline for classifying fraud, both with only base features and engineered features. The expectation was that the neural nets would perform (relatively) better with base features only, by learning their own features. Further measures could have been taken to improve neural net performance e.g. downsampling genuines and handling missing values more appropriately. NODE seems to do better than TabNet on base features only, suggesting it is better at generating complex features, but with engineered features, TabNet performs just as well. Both NODE and TabNet significantly outperform simple MLPs (with minimal tuning), but underperform XGBoost using the same basic preprocessing. 
