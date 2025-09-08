from app.bnn.redneuronal_bay.Layers.base_layer import BaseLayer
from app.bnn.redneuronal_bay.funcion_activacion import *
import numpy as np
import torch
from torch.autograd import Variable


class Sigmoid_Layer(BaseLayer):

    def __init__(self, input_dim, output_dim):
        """
        :input_dim : Dimension de entrada de la capa
        :output_dim: dimension de salida de la capa
        """
        BaseLayer.__init__(self, input_dim, output_dim, funcion_activacion=sigmoide)

    def derivative(self, x):
        """
        :x: Entrada de la funcion sigmoide.
        :regresa: Derivada de la sigmoide
        """
        x = x.detach().numpy()
        dv = np.multiply((1.0 / (1.0 + np.exp(-x))), (1 - (1.0 / (1.0 + np.exp(-x)))))
        return Variable(torch.FloatTensor(dv), requires_grad=True)


class Tanh_Layer(BaseLayer):

    def __init__(self, input_dim, output_dim):
        """
        :input_dim : Dimension de entrada de la capa
        :output_dim: dimension de salida de la capa
        """
        BaseLayer.__init__(self, input_dim, output_dim, funcion_activacion=tan_h)

    def derivative(self, x):
        """
        :x: Entrada de la funcion sigmoide.
        :regresa: Derivada de la tanh
        """
        # return (1 - np.square(x))
        return 1 - torch.square(x)
        # return (x)


class ReLU_Layer(BaseLayer):

    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: input dimension of layer
        :param output_dim: output dimension of layer
        """
        BaseLayer.__init__(self, input_dim, output_dim, funcion_activacion=ReLU)

    def derivative(self, x):
        """
        :x: Entrada de la funcion sigmoide.
        :regresa: Derivada de la Relu
        """
        x = x.detach().numpy()

        dv = np.clip(x > 0, 0.0, 1.0)
        return Variable(torch.FloatTensor(dv), requires_grad=True)


class LeakyReLU_Layer(BaseLayer):

    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: input dimension of layer
        :param output_dim: output dimension of layer
        """
        BaseLayer.__init__(self, input_dim, output_dim, funcion_activacion=Leaky_ReLU)

    def derivative(self, x):
        """
        :x: Entrada de la funcion sigmoide.
        :regresa: Derivada de la Relu con valor minimo de 0.01.
        """
        x = x.detach().numpy()

        dv = np.clip(x > 0, 0.01, 1.00)
        return Variable(torch.FloatTensor(dv), requires_grad=True)


class Softmax_Layer(BaseLayer):

    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: input dimension of layer
        :param output_dim: output dimension of layer
        """
        BaseLayer.__init__(self, input_dim, output_dim, funcion_activacion=softmax)

    def derivative(self, x):
        """
        :param x: Softmax activated input.
        :return: derivate of softmax function.
        """
        "pass"
        # print(x)
        x = x.detach().numpy()
        lis = np.diag(x[0])
        dim = lis.shape
        lis = np.zeros(dim)

        for i in range(len(x)):
            pred = x[i]
            dv = np.diag(x[i])  # Jacobiano
            for i in range(len(dv)):
                for j in range(len(dv)):
                    if i == j:
                        dv[i][j] = pred[i] * (1 - pred[i])  # dv[i][j] = x[i] * (1-x[j])
                    else:
                        dv[i][j] = -pred[i] * pred[j]
            for i in range(len(lis)):  # Para sumar todos las derivadas de cada muestra
                for j in range(len(lis[0])):
                    lis[i][j] = lis[i][j] + dv[i][j]

        lis = lis / (len(x))

        return Variable(torch.FloatTensor(lis), requires_grad=True)


class SoftmaxBay_Layer(BaseLayer):

    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: input dimension of layer
        :param output_dim: output dimension of layer
        """
        BaseLayer.__init__(self, input_dim, output_dim, funcion_activacion=softmax_Bay)

    def derivative(self, x):
        """
        :param x: Softmax activated input.
        :return: derivate of softmax function.
        """
        "pass"
        # print(x)
        x = x.detach().numpy()
        lis = np.diag(x[0])
        dim = lis.shape
        lis = np.zeros(dim)

        for i in range(len(x)):
            pred = x[i]
            dv = np.diag(x[i])  # Jacobiano
            for i in range(len(dv)):
                for j in range(len(dv)):
                    if i == j:
                        dv[i][j] = pred[i] * (1 - pred[i])  # dv[i][j] = x[i] * (1-x[j])
                    else:
                        dv[i][j] = -pred[i] * pred[j]
            for i in range(len(lis)):  # Para sumar todos las derivadas de cada muestra
                for j in range(len(lis[0])):
                    lis[i][j] = lis[i][j] + dv[i][j]

        lis = lis / (len(x))

        return Variable(torch.FloatTensor(lis), requires_grad=True)


class Log_Softmax_Layer(BaseLayer):

    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: input dimension of layer
        :param output_dim: output dimension of layer
        """
        BaseLayer.__init__(self, input_dim, output_dim, funcion_activacion=log_softmax)

    def derivative(self, x):
        """
        :param x: Softmax activated input.
        :return: derivate of softmax function.
        """
        "pass"
        # print(x)
        x = x.detach().numpy()
        lis = np.diag(x[0])
        dim = lis.shape
        lis = np.zeros(dim)

        for i in range(len(x)):
            pred = x[i]
            dv = np.diag(x[i])  # Jacobiano
            for i in range(len(dv)):
                for j in range(len(dv)):
                    if i == j:
                        dv[i][j] = pred[i] * (1 - pred[i])  # dv[i][j] = x[i] * (1-x[j])
                    else:
                        dv[i][j] = -pred[i] * pred[j]
            for i in range(len(lis)):  # Para sumar todos las derivadas de cada muestra
                for j in range(len(lis[0])):
                    lis[i][j] = lis[i][j] + dv[i][j]

        lis = lis / (len(x))

        return Variable(torch.FloatTensor(lis), requires_grad=True)
