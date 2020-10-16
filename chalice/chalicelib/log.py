#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging


# Logger.
LOGGER_ = None


def init(name='root'):
    """Configures a global logger.
​
    Return
    ------
    logger : logging.Logger
        Logger.
    """
    global LOGGER_

    if LOGGER_ is not None:
        return

    # Log format.
    formatter = logging.Formatter(
        '%(asctime)s :: '
        '%(levelname)s :: '
        '%(funcName)s :: '
        '%(lineno)d :: '
        '%(message)s')

    # Create handler and set format.
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Create logger.
    logger = logging.getLogger(name)
    logger.propagate = False

    # Set level
    logger.setLevel(logging.INFO)

    # Attatch handler
    logger.addHandler(handler)

    LOGGER_ = logger

    return
