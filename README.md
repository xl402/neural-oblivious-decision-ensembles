# Keras implementation of Neural Oblivious Decision Ensembles
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/xl402/node/node)


An implementation of <a href="https://arxiv.org/abs/1909.06312">NODE</a> - a differentiable oblivious decision tree using soft decision boundaries


<img src="https://imgur.com/EWA1sdj.png" width="500px"></img>


### Initial Setup
Create a Python 3 virtual environment and activate:
```
virtualenv -p python3 env
source ./env/bin/activate
```
Install requirements by running:
```
pip install -r requirements.txt
```
Then export project to python path:
```
export PYTHONPATH=$PATH_TO_REPO/node
```
To test the scripts, run `pytest` in the root directory, you may wish to
install `pytest` separately

### Usage
Below is an example of a 3 class classifier implemented with 5 layers of
decision tree ensemble, each of depth 3 with 100 estimators.
```python
import tensorflow as tf
from node.networks.model import NODE


model = NODE(n_layers=5,
	     n_trees=100,
	     tree_depth=3,
	     units=3,
	     link=tf.keras.activations.softmax)
x = tf.keras.Input(shape=10)
y = model(x)
print(y.shape)
# (None, 3)
```
