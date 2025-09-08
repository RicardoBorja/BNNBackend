from __future__ import division
from abc import abstractmethod, ABCMeta
import numpy as np
from app.bnn.redneuronal_bay.funcion_activacion import *
from app.bnn.redneuronal_bay import RedNeuBay
import torch
import math
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class BaseLayer:
    """
    Creo un objeto para las capas de la red
    """

    __metaclass__ = ABCMeta

    def __init__(self, input_dim, output_dim, funcion_activacion):
        """
        :input_dim: Input Dimension for this Layer
        :output_dim: Output Dimension for this Layer
        :transformFunction: Non-Linear transform puede ser aplicado en la salida ojo revisar si esta bien ???? o debo incluir como
        :                   entada
        """

        assert (
            input_dim > 0
            and isinstance(input_dim, int)
            and output_dim > 0
            and isinstance(output_dim, int)
        ), (
            "Las dimensiones de la entrrada y salida deben ser mayores a cero, "
            "Valores input_dim = {0} y output_dim = {1}".format(input_dim, output_dim)
        )

        torch.manual_seed(1234)
        #######torch.manual_seed(2468)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.funcion_activacion = funcion_activacion
        self.weights = Parameter(torch.Tensor(output_dim, input_dim))  # adicionado
        self.bias = Parameter(torch.Tensor(output_dim))  # adicionado

        # self.weights = Variable(torch.FloatTensor(self.__set_weights(self.input_dim, self.output_dim))) #origina

        self.weights = torch.nn.init.kaiming_uniform_(
            torch.Tensor(output_dim, input_dim), a=math.sqrt(5)
        )  # adicionado
        # self.weights = torch.nn.init.kaiming_normal_(torch.Tensor(output_dim, input_dim), a=0, mode='fan_out') #adicionado

        # -------------------------------------------------------------------------------
        # Si coloco mode= fan_in conserva la magnitud de la varianza de los pesos en el pase hacia adelante . La elección 'fan_out'conserva
        # las magnitudes en el pase hacia atrás. Podemos escribir lo siguiente.
        # ----------------------------------------------------------------------------------------------

        self.weights = self.weights.T  # adicionado
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
            self.weights
        )  # adicionado
        limite = 1 / math.sqrt(fan_in)  # adicionado
        self.bias = torch.nn.init.uniform_(self.bias, -limite, limite)  # adicionado
        self.bias = self.bias.T  # adicionado
        # self.weights = torch.nn.init.uniform_(torch.Tensor(output_dim, input_dim)) # adicionado

        # self.bias = Variable(torch.FloatTensor(np.ones((1,self.output_dim))))  # original

    # def __set_weights(self,input_dim, output_dim):
    #    np.random.seed(1234)
    #    return np.random.normal(0,1,output_dim*input_dim).reshape(input_dim,output_dim)

    @abstractmethod
    def derivative(self, x):
        pass

    def __str__(self):
        """
        Me anula la funcion de impresion de python.
        :regresa: capa detallada.
        """
        return (
            "{:>10s}{:>18d}\n".format("Input dimension", self.input_dim)
            + "{:>10s}{:>17d}\n".format("Output dimension", self.output_dim)
            + "{:>s}{:>18s}\n".format(
                "Activation Function", self.funcion_activacion.__name__
            )
            + "{:>10s}{:>23s}\n".format("Weights Shape", self.weights.shape)
        )
