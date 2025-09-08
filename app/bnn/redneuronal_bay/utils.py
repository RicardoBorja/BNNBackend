"""
Funciones que se utilicen para el modelo
"""

import sys
import numpy as np

np.random.seed(2312)
import torch
import math
from torch.nn.parameter import Parameter


def convert_1D(tensor):
    """
    Convierte de numpy array a array de 1D
    :tensor: Entrada de tipo "numpy array"
    :regresa: array de 1D
    """
    return np.ravel(tensor)


def x1_igual_x2(x, y):
    """
    Revisa que sea igual el numero de instancias en dos numpy array.
    :x: numpy array
    :y: numpy array
    :regresa: None
    """
    assert (
        x.shape[0] == y.shape[0]
    ), "Instancias de los dos conjuntos no tienen las mismas dimensiones (shape)"


def cambia_labels(targets):
    """
    Cambia las etiquetas de acuerdo a los datos requeridos por la red
    Por ejemplo de: targets = [1,0,1]
    Cambia a : targets=  [[0,1],[1,0],[0,1]]
    :targets:Es un "numpy array" de dimension 1D.
    :regresa: numpy array donde esta activa la etiqueta de la clase
    """

    targets = targets.ravel()
    numclass = len(np.unique(targets))
    new_targets = []
    for label in targets:
        temp = np.zeros(numclass)
        temp[int(label)] = 1
        new_targets.append(temp)

    return np.array(new_targets)


def genera_batches(data_size, batch_size):
    """
    Genera conjuntos de datos basados en batch size.
    :data_size: Numero total de instancias de los datos.
    :batch_size: Cantidad de instancias en cada conjunto (de tipo int)

    """
    assert isinstance(
        batch_size, int
    ), "batch_size debe ser un valor entero, {0} error.".format(type(batch_size))
    assert batch_size > 0, "batch_size debe ser mayor a cero"
    ind = np.array([i for i in range(data_size)])
    np.random.shuffle(ind)
    temp = []
    for i in range(0, data_size, batch_size):
        temp.append(ind[i : i + batch_size])
    return np.array(temp)


def act_Weig_bias(Rn):

    # torch.manual_seed(1234)
    for i in range(len(Rn.layersObject)):

        input_dim = Rn.layersObject[i].weights.shape[0]
        output_dim = Rn.layersObject[i].weights.shape[1]
        weights = Parameter(torch.Tensor(output_dim, input_dim))
        bias = Parameter(torch.Tensor(output_dim))

        weights = torch.nn.init.kaiming_uniform_(torch.Tensor(output_dim, input_dim))
        Rn.layersObject[i].weights = weights.T
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        limite = 1 / math.sqrt(fan_in)
        bias = torch.nn.init.uniform_(bias, -limite, limite)
        Rn.layersObject[i].bias = bias.T


def progreso_epo(
    ite_act, total, total_epoch, act_epoch, prefix="", decimales=0, barLength=35
):
    """
    Imprime el progreso de cada epoch.

    :ite_act: iteration actual
    :total: total de iteraciones
    :total_epoch: total epochs a realizar
    :act_epoch: epoch actual
    :prefix: no requerido(Progreso predeterminado)
    :decimales: numero positivo de decimales en el porcentaje (Int)
    :barLength: longitud de la barra
    :regresa: None
    """
    formatStr = "{0:." + str(decimales) + "f}"
    porcent = formatStr.format(100 * (ite_act / float(total)))
    fullLength = int(round(barLength * ite_act / float(total)))
    bar = ">" * fullLength + "-" * (barLength - fullLength)
    sys.stdout.write(
        "\repoch %s/%s %s %s %s%s" % (act_epoch, total_epoch, prefix, bar, porcent, "%")
    ),
    if ite_act == total:
        sys.stdout.write("\n")
    sys.stdout.flush()
