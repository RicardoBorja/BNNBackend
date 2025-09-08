from __future__ import division, absolute_import
import app.bnn.redneuronal_bay.utils as utils
import numpy as np

"""
This file contains implementation of common metric functions for
classification and regression.
"""


"Classification"


def accuracy_score(true, predicted):
    """
    :true: vector que contiene todas las clases reales
    :predicted: vector que contiene todas las clases predichas
    :regresa: accuracy del probllema de clasificacion
    """

    true = utils.convert_1D(true)
    predicted = np.round(predicted)
    predicted = utils.convert_1D(predicted)
    for i in range(len(predicted)):
        if predicted[i] == -1:
            predicted[i] = 0
        else:
            predicted[i] = predicted[i]
    assert (
        true.shape == predicted.shape
    ), "true y predicted no tienen las mismas dimensiones"
    return (
        true.shape[0] - np.count_nonzero(np.subtract(true, predicted))
    ) / true.shape[0]


"Regression"


def mean_absolute_error(true, predicted):
    """
    :regresa: mean absolute error.
    """
    true = utils.convert_1D(true)
    predicted = utils.convert_1D(predicted)
    assert true.shape == predicted.shape, "true and pred dimensions do not match."
    return np.sum(np.fabs(np.subtract(true, predicted))) / true.shape[0]


def mean_squared_error(true, predicted):
    """
    :regresa: mean squared error.
    """
    true = utils.convert_1D(true)
    predicted = utils.convert_1D(predicted)
    assert (
        true.shape == predicted.shape
    ), "true y predicted no tienen las mismas dimensiones"
    return np.sum(np.square(np.subtract(true, predicted))) / true.shape[0]


def r2_score(true, predicted):
    """
    :regresa    : R-squared(coeficiente de determinacion)
    """
    true = utils.convert_1D(true)
    predicted = utils.convert_1D(predicted)
    assert (
        true.shape == predicted.shape
    ), "true y redicted no tienen las mismas dimensiones"
    numerator = np.sum(np.square(np.subtract(true, predicted)))
    denominator = np.sum(np.square(np.subtract(true, np.mean(true))))
    return 1 - (numerator / denominator)
