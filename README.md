
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
    X, y, test_size=0.1, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=.001)

model.fit(
    X_train, y_train,
    early_stopping_rounds=10,
    eval_set=[(X_test, y_test)])
```

    [0]	validation_0-error:0.070175
    Will train until validation_0-error hasn't improved in 10 rounds.
    [1]	validation_0-error:0.070175
    [2]	validation_0-error:0.070175
    [3]	validation_0-error:0.070175
    [4]	validation_0-error:0.070175
    [5]	validation_0-error:0.070175
    [6]	validation_0-error:0.070175
    [7]	validation_0-error:0.070175
    [8]	validation_0-error:0.070175
    [9]	validation_0-error:0.070175
    [10]	validation_0-error:0.070175
    Stopping. Best iteration:
    [0]	validation_0-error:0.070175






    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0,
           learning_rate=0.001, max_delta_step=0, max_depth=3,
           min_child_weight=1, missing=None, n_estimators=500, n_jobs=1,
           nthread=None, objective='binary:logistic', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=None, subsample=1, verbosity=1)




```python
MODEL_FOLDER_PATH = os.path.expanduser(
    '~/chalice_xgboost/chalicelib/models/')

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

    Number of trees: 11
    Number of top-n best trees: 1



```python

```

Bellow we show how the JSON dump of trees


```python
model_json
```




    [{'nodeid': 0,
      'depth': 0,
      'split': 'mean concave points',
      'split_condition': 0.0512799993,
      'yes': 1,
      'no': 2,
      'missing': 1,
      'children': [{'nodeid': 1,
        'depth': 1,
        'split': 'worst radius',
        'split_condition': 16.8299999,
        'yes': 3,
        'no': 4,
        'missing': 3,
        'children': [{'nodeid': 3,
          'depth': 2,
          'split': 'radius error',
          'split_condition': 0.572100043,
          'yes': 7,
          'no': 8,
          'missing': 7,
          'children': [{'nodeid': 7, 'leaf': 0.00191752589},
           {'nodeid': 8, 'leaf': 0}]},
         {'nodeid': 4,
          'depth': 2,
          'split': 'mean texture',
          'split_condition': 18.6800003,
          'yes': 9,
          'no': 10,
          'missing': 10,
          'children': [{'nodeid': 9, 'leaf': 0.00100000005},
           {'nodeid': 10, 'leaf': -0.00125000009}]}]},
       {'nodeid': 2,
        'depth': 1,
        'split': 'worst concave points',
        'split_condition': 0.14655,
        'yes': 5,
        'no': 6,
        'missing': 6,
        'children': [{'nodeid': 5,
          'depth': 2,
          'split': 'worst perimeter',
          'split_condition': 115.25,
          'yes': 11,
          'no': 12,
          'missing': 12,
          'children': [{'nodeid': 11, 'leaf': 0.00106666679},
           {'nodeid': 12, 'leaf': -0.00155555562}]},
         {'nodeid': 6,
          'depth': 2,
          'split': 'concavity error',
          'split_condition': 0.112849995,
          'yes': 13,
          'no': 14,
          'missing': 14,
          'children': [{'nodeid': 13, 'leaf': -0.00192546588},
           {'nodeid': 14, 'leaf': 0}]}]}]}]



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




    0    0.500479
    1    0.499519
    2    0.499519
    3    0.500479
    4    0.500479
    dtype: float64



Get model scores via `xgboost.XGBClassifier` object


```python
y_scores_model = pd.Series(model.predict_proba(
    X_test
)[:, 1])

y_scores_model.head()
```




    0    0.500479
    1    0.499519
    2    0.499519
    3    0.500479
    4    0.500479
    dtype: float32



In this section we have show how to create a XGBoost model, saved it as a JSON set of trees and how to get model probabilities scores using pure python code.

# Step 3: Use Chalice to deploy your model.

In this section, we will use Chalice in order to deploy our serverless infrastructure in AWS. You will first need to set up your AWS credentials as shown in the following [link](https://github.com/aws/chalice). Once your AWS account and your credentials are all in place, we will only need to:

- Create a new python environment and install the `chalice` package.
- Clone the minimalist chalice project that contains our XGboost model dump.
- Local test and deploy to production.

## Create a new python environment and install the `chalice` package.

First, create the python environment. You can use your favorite environment management tool, in this example I'll use conda:
```
$ conda create --name chalice_xgboost python=3.7.3
$ conda activate chalice_xgboost
(chalice_xgboost) $
```

Then install chalice package:
```
(chalice_xgboost) $ pip install chalice
```

With all set-up, we'll go straight and clone a GitHub project in which we already have all set-up for you in order to be able to best show a minimalistic example of how to get model scores.


## Clone the minimalist chalice project that contains our XGboost model dump.
```
(chalice_xgboost) $ git clone git@github.com:RaulSanchezVazquez/chalice_xgboost.git
```

## Local test and deploy to production.
```
(chalice_xgboost) LuisSanchez-MBP:chalice_xgb lsanchez$ chalice local
Serving on http://127.0.0.1:8000
```



```python
x = X_test.iloc[9].to_dict()
x
```




    {'mean radius': 13.9,
     'mean texture': 16.62,
     'mean perimeter': 88.97,
     'mean area': 599.4,
     'mean smoothness': nan,
     'mean compactness': 0.05319,
     'mean concavity': 0.02224,
     'mean concave points': 0.01339,
     'mean symmetry': 0.1813,
     'mean fractal dimension': 0.05536,
     'radius error': 0.1555,
     'texture error': 0.5762,
     'perimeter error': 1.392,
     'area error': 14.03,
     'smoothness error': 0.003308,
     'compactness error': 0.01315,
     'concavity error': 0.009904,
     'concave points error': 0.004832,
     'symmetry error': 0.01316,
     'fractal dimension error': 0.002095,
     'worst radius': 15.14,
     'worst texture': 21.8,
     'worst perimeter': 101.2,
     'worst area': 718.9,
     'worst smoothness': 0.09384,
     'worst compactness': 0.2006,
     'worst concavity': 0.1384,
     'worst concave points': 0.06222,
     'worst symmetry': 0.2679,
     'worst fractal dimension': 0.07698}




```python
import os
import json
import urllib3

http = urllib3.PoolManager()

ENDPOINT = 'http://127.0.0.1:8000'

body = {"x": x}

response = http.request(
     'POST',
     ENDPOINT + '/predict_proba',
     body=json.dumps(body).encode('utf-8'),
     headers={'Content-Type': 'application/json'})

response = json.loads(
    response.data.decode('utf-8'))

response
```




    {'response': {'y_score': [0.4995206186743867, 0.5004793813256133]}}




```python

```
