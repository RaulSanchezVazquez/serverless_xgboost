# Serverless XGBoost

There are many ways that data scientists can contribute to the projects or companies they work with. Arguably, one of the biggest contributions that a data scientist can make is to deploy a model that makes inference in real-time on an online setting. 

When it comes to deploying models, such as XGBoost models, there are various tools that allow you to achieve the same and ultimate goal: online inference. This tutorial will illustrate my favorite choice in the context of XGBoost, that consists of a small set of technologies easy to understand, very reliable, secure, scalable, and affordable.

The reasons for me that made of this approach my favorite is mainly due to the following advantages:

- There is no need to install any package in the production environment
- It is very affordable for all project sizes.
- It is powered by cloud solutions that made it very scalable.

# Fit and Deploy a XGBoost Binary Classifier.

## Step 1: Fit a XGBoost Binary Classifier.

In this section we'll fit a binary XGBoost classifier to the Breast Cancer dataset.

In the dataset, we will artificially insert a few `NaN`. This will allow us to show that the pure python code correctly handles missing values. Also, we will use `pandas.DataFrame` so that the split nodes in the XGBoost trees contain the feature names, which will make our tree structure a bit more human readable.
And last, we will use early stopping in order to show how to correctly fetch scores from the top-N trees.

The resulting model will be saved in JSON format insead to the classic pickle format. By saving the model as a JSON instead of Pickle format will allow us to skip the XGBoost instalation in our production setting.


```python
import os
import pandas as pd
import xgboost as xgb
import json

from sklearn import datasets
from sklearn.model_selection import train_test_split

dataset = datasets.load_breast_cancer()
X = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names)
y = dataset.target
```


```python
nan_count_original = X.isnull().sum().sum()
for col in X.columns:
    X.loc[
        X[col] <= X[col].quantile(.01),
        col] = pd.np.nan
nan_count_artificial = X.isnull().sum().sum()

print('NaN values count in original dataset: %s' % nan_count_original)
print('NaN values count in artificial dataset: %s' % nan_count_artificial)
```

    NaN values count in original dataset: 0
    NaN values count in artificial dataset: 222


    /Users/lsanchez/anaconda3/envs/py37/lib/python3.7/site-packages/ipykernel_launcher.py:5: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead
      """



```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=.001)

model.fit(
    X_train, y_train,
    early_stopping_rounds=10,
    eval_set=[(X_test, y_test)])
```

    [0]	validation_0-error:0.103448
    Will train until validation_0-error hasn't improved in 10 rounds.
    [1]	validation_0-error:0.068966
    [2]	validation_0-error:0.103448
    [3]	validation_0-error:0.103448
    [4]	validation_0-error:0.103448
    [5]	validation_0-error:0.103448
    [6]	validation_0-error:0.103448
    [7]	validation_0-error:0.103448
    [8]	validation_0-error:0.103448
    [9]	validation_0-error:0.103448
    [10]	validation_0-error:0.103448
    [11]	validation_0-error:0.103448
    Stopping. Best iteration:
    [1]	validation_0-error:0.068966
    





    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0,
           learning_rate=0.001, max_delta_step=0, max_depth=3,
           min_child_weight=1, missing=None, n_estimators=500, n_jobs=1,
           nthread=None, objective='binary:logistic', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=None, subsample=1, verbosity=1)




```python
MODEL_FOLDER_PATH = os.path.expanduser(
    '~/chalice_xgb/chalicelib/models/')

MODEL_FILE_PATH = os.path.join(
    MODEL_FOLDER_PATH, 'xgb.json')

# Make folder path
os.makedirs(
    MODEL_FOLDER_PATH,
    exist_ok=True)

# Save dump of trees as .json
model._Booster.dump_model(
    MODEL_FILE_PATH,
    dump_format='json')

# Open file just keep the best n-trees.
with open(MODEL_FILE_PATH, 'r') as f:
    model_json = json.loads(f.read())
print('Number of trees: %s' % len(model_json))

model_json = model_json[:model.best_ntree_limit]
print('Number of top-n best trees: %s' % len(model_json))

with open(MODEL_FILE_PATH, 'w') as f:
    f.write(json.dumps(model_json))
```

    Number of trees: 12
    Number of top-n best trees: 2


Bellow we show how the JSON dump of trees


```python
model_json
```




    [{'nodeid': 0,
      'depth': 0,
      'split': 'worst radius',
      'split_condition': 16.7950001,
      'yes': 1,
      'no': 2,
      'missing': 1,
      'children': [{'nodeid': 1,
        'depth': 1,
        'split': 'worst concave points',
        'split_condition': 0.160299987,
        'yes': 3,
        'no': 4,
        'missing': 3,
        'children': [{'nodeid': 3,
          'depth': 2,
          'split': 'worst concave points',
          'split_condition': 0.135800004,
          'yes': 7,
          'no': 8,
          'missing': 7,
          'children': [{'nodeid': 7, 'leaf': 0.00191250013},
           {'nodeid': 8, 'leaf': 0.000482758624}]},
         {'nodeid': 4, 'leaf': -0.00147826097}]},
       {'nodeid': 2,
        'depth': 1,
        'split': 'mean texture',
        'split_condition': 16.1100006,
        'yes': 5,
        'no': 6,
        'missing': 6,
        'children': [{'nodeid': 5,
          'depth': 2,
          'split': 'mean concavity',
          'split_condition': 0.119199999,
          'yes': 9,
          'no': 10,
          'missing': 10,
          'children': [{'nodeid': 9, 'leaf': 0.00138461543},
           {'nodeid': 10, 'leaf': -0.00120000006}]},
         {'nodeid': 6,
          'depth': 2,
          'split': 'worst concavity',
          'split_condition': 0.190699995,
          'yes': 11,
          'no': 12,
          'missing': 12,
          'children': [{'nodeid': 11, 'leaf': -0.000222222239},
           {'nodeid': 12, 'leaf': -0.00195121963}]}]}]},
     {'nodeid': 0,
      'depth': 0,
      'split': 'worst concave points',
      'split_condition': 0.142349988,
      'yes': 1,
      'no': 2,
      'missing': 1,
      'children': [{'nodeid': 1,
        'depth': 1,
        'split': 'worst area',
        'split_condition': 957.450012,
        'yes': 3,
        'no': 4,
        'missing': 3,
        'children': [{'nodeid': 3,
          'depth': 2,
          'split': 'area error',
          'split_condition': 35.4350014,
          'yes': 7,
          'no': 8,
          'missing': 7,
          'children': [{'nodeid': 7, 'leaf': 0.0019098405},
           {'nodeid': 8, 'leaf': 0.000713152287}]},
         {'nodeid': 4,
          'depth': 2,
          'split': 'worst fractal dimension',
          'split_condition': 0.0649200007,
          'yes': 9,
          'no': 10,
          'missing': 10,
          'children': [{'nodeid': 9, 'leaf': 0.000222551418},
           {'nodeid': 10, 'leaf': -0.0016349256}]}]},
       {'nodeid': 2,
        'depth': 1,
        'split': 'worst radius',
        'split_condition': 15.4099998,
        'yes': 5,
        'no': 6,
        'missing': 6,
        'children': [{'nodeid': 5,
          'depth': 2,
          'split': 'mean smoothness',
          'split_condition': 0.108150005,
          'yes': 11,
          'no': 12,
          'missing': 12,
          'children': [{'nodeid': 11, 'leaf': 0.00133317523},
           {'nodeid': 12, 'leaf': -0.00127196533}]},
         {'nodeid': 6,
          'depth': 2,
          'split': 'radius error',
          'split_condition': 0.241250008,
          'yes': 13,
          'no': 14,
          'missing': 14,
          'children': [{'nodeid': 13, 'leaf': -0.000399500976},
           {'nodeid': 14, 'leaf': -0.00192462502}]}]}]}]



## **Step 2**: Get model scores from the model json dump.

The following two methods will allow us to fetch model scores.

The first method, namely `get_tree_leaf()` will allow us to fetch a single tree leaf score value.
And the second method: `binary_predict_proba()` will allow us to iteratively fetch model scores, sum them up and get the final probability score.


```python
import math

def get_tree_leaf(node, x):
    """Get tree leaf score.
    
    Each node contains childres that are composed of aditiona nodes.
    Final nodes with no children are the leaves.
    
    Parameters
    -----------
    node: dict.
        Node XGB dictionary.
    x: dict.
        Dictionary containing feature names and feature values.
    
    Return
    -------
    score: float.
        Leaf score.
    """

    if 'leaf' in node:
        # If the key leaf is found, the stop recurrency.
        score = node['leaf']
        return score
    else:
        # Get current split feature value
        x_f_val = x[node['split']]

        # Get next node.
        if str(x_f_val) == 'nan':
            # if split feature value is nan.
            next_node_id = node['missing']
        elif x_f_val < node['split_condition']:
            # Split condition is true.
            next_node_id = node['yes']
        else:
            # Split condition is false.
            next_node_id = node['no']

        # Dig down to the next node.
        for children in node['children']:
            if children['nodeid'] == next_node_id:
                return get_tree_leaf(children, x)


def binary_predict_proba(x, model_json):
    """Get score of a binary xgboost classifier.
    
    Parameters
    ----------
    x: dict.
        Dictionary containing feature names and feature values.
    
    model_json: dict.
        Dump of xgboost trees as json.
    
    Returns
    -------
    y_score: list
        Scores of the negative and positve class.
    """
    
    # Get tree leafs.
    tree_leaf_scores = []
    for tree in model_json:
        leaf_score = get_tree_leaf(
            node=tree,
            x=x)
        tree_leaf_scores.append(leaf_score)

    # Get logits.
    logit = sum(tree_leaf_scores)
    
    # Compute logistic function
    pos_class_probability = 1 / (1 + math.exp(-logit))

    # Get negative and positive class probabilities.
    y_score = [1 - pos_class_probability, pos_class_probability]
    
    return y_score
```

Manually get model scores from the JSON dump.


```python
import json

# Model object
with open(MODEL_FILE_PATH, 'r') as f:
    model_json = json.loads(f.read())

y_scores_json = pd.Series([
    binary_predict_proba(x.to_dict(), model_json)[1]
    for _, x in X_test.iterrows()])

y_scores_json.head()
```




    0    0.500956
    1    0.499031
    2    0.499031
    3    0.500956
    4    0.500956
    dtype: float64



Get model scores via `xgboost.XGBClassifier` object


```python
y_scores_model = pd.Series(model.predict_proba(
    X_test
)[:, 1])

y_scores_model.head()
```




    0    0.500956
    1    0.499031
    2    0.499031
    3    0.500956
    4    0.500956
    dtype: float32



In this section we have show how to create a XGBoost model, saved it as a JSON set of trees and how to get model probabilities scores using pure python code.

# Step 3: Use Chalice to deploy your model.

In this section we will use Chalice in order to deploy our serverless infraestructure in AWS. You will first need to set up your AWS credentials as show in the following [link](https://github.com/aws/chalice). 
Once your AWS account and your credentials are all in place, we will only need to:

- Create a new python environment.
- Install the `chalice` package.
- Build a chalice project, which contain our XGboost JSON dump and python script to fetch scores.
- Test locally and deploy to production.

So first create the python environment, you can use your vaborite environment management, in this example I'll use conda:
```
$ conda create --name chalice_xgboost python=3.7.3
$ conda activate chalice_xgboost
(chalice_xgboost) $
```

Then install chalice package:
```
(chalice_xgboost) $ pip install chalice
```

to be continued...


```python

```
