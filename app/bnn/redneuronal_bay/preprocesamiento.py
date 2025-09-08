import numpy as np


class label_encoder:
    """
    Codifica los elementos de una matriz desde el rango 0 hasta las clases numericas -1 (ultima)
    """

    def __init__(self):
        self._classes = None
        self.transformation = None

    def fit(self, array):
        """
        :array: entrada numpy array 1D
        :regresa: None
        """
        assert array.ndim == 1, "requiere un array de 1D"
        self._classes, self._transformation = np.unique(array, return_inverse=True)

    def transform(self, array):
        """
        :array: numpy array a ser transformado
        :regresa: Matriz codificada  - etiquetada.
        """
        assert array.ndim == 1, "requiere array 1D"
        return np.searchsorted(self._classes, array)

    def inverse_transform(self, array):
        """
        :array: numpy array a ser transformado.
        :regresa: Matriz decodificada
        """
        assert array.ndim == 1, "Requiere array 1D"
        return self._classes[array]
