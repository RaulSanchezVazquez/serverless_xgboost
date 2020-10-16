#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
