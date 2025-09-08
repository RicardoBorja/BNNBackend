from __future__ import division
import numpy as np
import torch
import torch.nn as nn

"""
Definicion de las funciones de perdida (Loss).
"""


def SSE(true, predicted):
    # Error cuadrático medio
    # print(true.dtype) #Float 32
    # print(true.shape)
    # print(predicted)
    # print(predicted.dtype) # float 32 requiere grad
    Loss = nn.MSELoss()

    return Loss(predicted, true)


def cross_entropy(true, predicted):
    # print(true)   #Entropia cruzada - ojo va con la softmax como activacion final
    # print(true.dtype)
    # print(true.shape)
    # print(predicted)
    # print(predicted.dtype)
    # print(predicted.shape)
    # predicted = torch.LongTensor(predicted)
    # return (F.nll_loss(F.log_softmax(true, 1), predicted))
    Loss = nn.CrossEntropyLoss()
    return Loss(predicted, true)


def NLLLoss(true, predicted):
    # print(true)  # Logaritmico negativo - ojo va con la logsoftmax como activacion final
    # print(true.dtype) # LongTensor
    # print(true.shape)
    # print(predicted)
    # print(predicted.dtype) # FloatTensor 32
    # print(predicted.shape)
    # predicted = torch.LongTensor(predicted)
    Loss = nn.NLLLoss()
    return Loss(predicted, true)


def BCELoss(true, predicted):
    # print(true)  # Cross entropy binario - ojo va con la sigmoide como activacion final - solo se utiliza si quiero una etiqueta por cada
    # valor
    true = np.float32(true)
    true = torch.tensor(
        true
    )  # Porque la entrada debe ser flotante y no int como lo es true
    # print(predicted)
    Loss = nn.BCELoss()
    return Loss(predicted, true)
