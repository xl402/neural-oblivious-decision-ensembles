# Tensorflow Implementation of Neural Oblivious Decision Ensembles
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/xl402/node/node)


An implementation of <a href="https://arxiv.org/abs/1909.06312">NODE</a>, a differentiable oblivious decision tree using soft decision boundaries


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
