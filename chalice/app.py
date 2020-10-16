#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time

from chalice import Chalice

from chalicelib import config
from chalicelib import log
from chalicelib import model
from chalicelib import xgb


# Create the chalice app.
app = Chalice(app_name='kpay_fraud_service')

# Initialize the logger.
log.init()


# Index.
@app.route('/')
def index():
    """Get model version
    """
    response = {'version': config.VERSION}
    log.LOGGER_.info('Response: %s' % response)

    return response


# Get model score.
@app.route('/predict_proba', methods=['POST'])
def get_decision():
    """Get model score.
    """
    # Get body
    body = app.current_request.json_body

    # Get features from the request
    x = body['x']
    log.LOGGER_.info('x: %s' % x)

    # Load the model.
    model_json = model.get()
    log.LOGGER_.info('Model loaded: %s Trees' % len(model_json))

    # Get decision thresholds.
    y_score = xgb.binary_predict_proba(x, model_json)
    log.LOGGER_.info('y_score: %s' % y_score)

    # Get response
    response = {'y_score': y_score}
    log.LOGGER_.info('response: %s' % response)

    return {'response': response}
