#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os


# Model filepath.
MODEL_FILE_PATH = os.path.join(
    os.path.dirname(__file__), 'models/xgb.json')


def get():
    """Fech XGBoost model as dict.

    Returns
    --------
    model_json: dict.
        The XGBoost model object.
    """

    # Open file just keep the best n-trees.
    with open(MODEL_FILE_PATH, 'r') as f:
        model_json = json.loads(f.read())

    return model_json
