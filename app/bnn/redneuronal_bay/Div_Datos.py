"""
Toma la base de datos y la divide en entrenamiento y validacion, además de separar los datos de la etiqueta.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# np.random.seed(2312)


def trat_Dat(df_cla=False, X_cla=False, Y_cla=False, batch_size=64, test_size=0.20):
    """
    Realiza las divisiones necesarias en caso se trabaje con una base de datos. Debe ingresarse dos archivos o uno (predictores y
     etiquetas)que contengan todos los datos de entrenamiento y validaciòn. La división lo realizará internamiente (train y valid).
    """
    # CUDA = torch.cuda.is_available()
    CUDA = False
    df_cla = pd.DataFrame(df_cla)
    # print(len(df_cla))

    # ----------------------------------------------------------------------------
    # Para separar los predictores de la variable respuesta
    # La entrada puede ser un solo archivo tipo Dataframe de pandas o ya dividido X y Y de entrenamiento y test
    # ----------------------------------------------------------------------------
    if X_cla == False and Y_cla == False:
        array = df_cla.values
        predictors = df_cla.shape[1]
        X_cla = array[:, 0 : (predictors - 1)]
        Y_cla = array[:, (predictors - 1)]
    elif df_cla == False:
        X_cla = np.asarray(X_cla)
        Y_cla = np.asarray(Y_cla)

    # ----------------------------------------------------------------------------
    # Para el caso que la entrada fuera una sola base y se debe dividir en entrenamiento y test.
    # Debe ingresarse un archivo de datos y otro de etiquetas
    # ----------------------------------------------------------------------------

    # test_size = 0.20  # Porcentaje para division del test
    seed = 7
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_cla, Y_cla, test_size=test_size, random_state=seed
    )

    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # Para convertir la base de datos en una iterable y compatible con el proceso para imagenes
    # -----------------------------------------------------------------------------

    x = Variable(torch.FloatTensor(X_train), requires_grad=True)
    y = Variable(torch.LongTensor(Y_train))

    class NumbersDataset(Dataset):
        def __init__(self):
            self.inputs = list(x)
            self.targets = list(y)

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return self.inputs[idx], self.targets[idx]

    dataset = NumbersDataset()  # Nuevo nombre de la base de datos

    # -----------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # División de datos de acuerdo al batch size escogido. Unicamente del entrenamiento
    # ya que del de test nos sirve la fase de division inicial (X_test y Y_test)
    # -----------------------------------------------------------------------------

    # batch_size = 64 #64 #128
    if batch_size == None:
        batch_size = len(dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=CUDA
    )
    # ------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # Descriptivo de los datos
    # -----------------------------------------------------------------------------

    # Cantidad de elementos deacuerdo a cada etiqueta
    print("-----------------------------------")
    print("Frecuencia total por etiqueta")
    class_counts = df_cla.groupby("class").size()
    print(df_cla.groupby("class").size())
    print("-----------------------------------")
    # Numero de imagenes y dimension (i,j)
    print("---------------Train---------------")
    print("Numero de datos =", len(dataset))
    print("Dimension de los datos=", len(dataset), "X", X_cla.shape[1])
    print("-----------------------------------")

    # Save frequency per tag to results.json
    results_path = os.path.join("app", "bnn", "output", "results.json")

    # Convert class counts to dictionary format
    class_counts_dict = {
        f"class_{int(idx)}": int(count) for idx, count in class_counts.items()
    }

    # Create or update results.json
    try:
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
        else:
            results = {}

        # Add or update the class frequency data
        if "overall_class_frequency" not in results:
            results["overall_class_frequency"] = {}

        results["overall_class_frequency"] = class_counts_dict

        # Ensure directory exists
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        # Save updated results
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

    except Exception as e:
        print(f"Error saving overall class frequency to results.json: {e}")

    # -----------------------------------------------------------------------------

    return train_loader, X_test, Y_test


def trat_Imag(train_set, test_set=False, batch_size=64, test_size=0.20):
    """
    Realiza las divisiones necesarias en caso se trabaje con una base de imagenes.
    Debe tener un input de la base de entrenamiento y validación por separado.
    La fase test es independiente.

    """

    # CUDA = torch.cuda.is_available()
    CUDA = False
    # -----------------------------------------------------------------------------
    # División de datos de acuerdo al batch size escogido
    # -----------------------------------------------------------------------------

    if test_set == False:
        X_test = 0
        Y_test = 0
    else:
        batch_size = len(
            test_set
        )  # 10000 # Para que pueda quedar un solo archivo test tomo toda la longitud de datos
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=False, pin_memory=CUDA
        )  # ojo el pin es solo en caso que cuda es == true
        for inputs, targets in test_loader:
            X_test, Y_test = inputs, targets

    if batch_size == None:
        batch_size = len(train_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=CUDA
    )
    # ------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # Descriptivo de los datos
    # -----------------------------------------------------------------------------
    # Numero de imagenes y dimension (i,j)
    print("---------------Train---------------")
    Nyd = train_set.train_data.shape
    print("Numero de imagenes =", Nyd[0])
    print("Dimension de las imagenes=", Nyd[1], "X", Nyd[2])
    print("-----------------------------------")

    print("-----------------------------------")
    # Cantidad de elementos deacuerdo a cada etiqueta
    rr = train_set.targets.data.numpy()
    rr = pd.DataFrame(rr, columns=["Labels"])
    num_lab = rr.groupby("Labels").size()
    print("Frecuencia total por etiqueta")
    print(num_lab)
    print("-----------------------------------")

    # Save frecuencia por etiqueta to results.json
    results_path = os.path.join("app", "bnn", "output", "results.json")

    # Convert frequency data to dictionary format
    freq_dict = {f"class_{int(idx)}": int(count) for idx, count in num_lab.items()}

    try:
        # Read existing results if file exists
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
        else:
            results = {}

        # Add or update the image class frequency data
        if "image_class_frequency" not in results:
            results["image_class_frequency"] = {}

        results["image_class_frequency"] = freq_dict

        # Ensure directory exists
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        # Save updated results
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

    except Exception as e:
        print(f"Error saving image class frequency to results.json: {e}")

    # -----------------------------------------------------------------------------

    return train_loader, X_test, Y_test


def cv_prepros(df_cla=False, X_cla=False, Y_cla=False):
    """
    Realiza solo devuelve el dataset en formato iterable y mezclado, listo para que se hagan las divisiones para el entrenamiento
     y validación.
    """
    df_cla = pd.DataFrame(df_cla)
    # print(len(df_cla))
    # ----------------------------------------------------------------------------
    # Para separar los predictores de la variable respuesta
    # La entrada puede ser un solo archivo tipo Dataframe de pandas o ya dividido X y Y de entrenamiento y test
    # ----------------------------------------------------------------------------
    if X_cla == False and Y_cla == False:
        array = df_cla.values
        predictors = df_cla.shape[1]
        X_cla = array[:, 0 : (predictors - 1)]
        Y_cla = array[:, (predictors - 1)]
        X_cla, Y_cla = shuffle(
            X_cla, Y_cla
        )  # Me baraja X_cla y Y_cla en los mismos indices
    elif df_cla == False:
        X_cla = np.asarray(X_cla)
        Y_cla = np.asarray(Y_cla)
        X_cla, Y_cla = shuffle(
            X_cla, Y_cla
        )  # Me baraja X_cla y Y_cla en los mismos indices

    return X_cla, Y_cla


def cv_trat_Dat(X, Y, batch_size, Kfold, inicio, tra_val, ite):

    # CUDA = torch.cuda.is_available()
    CUDA = False
    # -----------------------------------------------------------------------------
    # División de datos de acuerdo al CV que toca y al batch size escogido.

    # Para escoger los datos en cada cv
    if (ite + 1) == Kfold:
        fin = len(X) - 1
    else:
        fin = inicio + tra_val

    # test_set = (dataset[inicio:fin])
    X_te = X[inicio:fin, :]
    Y_te = Y[inicio:fin]

    X_tr = np.delete(X, np.s_[inicio:fin], axis=0)
    Y_tr = np.delete(Y, np.s_[inicio:fin], axis=0)

    # -----------------------------------------------------------------------------
    # Convierto los datos de entrenamiento en iterable

    x_t = Variable(torch.FloatTensor(X_tr), requires_grad=True)
    y_t = Variable(torch.LongTensor(Y_tr))

    class NumbersDataset(Dataset):
        def __init__(self):
            self.inputs = list(x_t)
            self.targets = list(y_t)

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return self.inputs[idx], self.targets[idx]

    data = NumbersDataset()  # Nuevo nombre de la base de datos

    if batch_size == None:
        batch_size = len(data)

    train_loader = torch.utils.data.DataLoader(
        dataset=data, batch_size=batch_size, shuffle=True, pin_memory=CUDA
    )
    # ------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------
    # Descriptivo de los datos
    # -----------------------------------------------------------------------------

    # Numero de datos - dimension (i,j)
    print("---------------Train---------------")
    Nyd = X_tr.shape
    print("Dimension =", Nyd[0], "X", (Nyd[1] + 1), "- en cv=", (ite + 1))
    print("-----------------------------------")

    print("-----------------------------------")
    # Cantidad de elementos deacuerdo a cada etiqueta
    rr = np.unique(y_t, return_counts=True)
    rr = np.asarray(rr)
    num_lab = pd.DataFrame(rr[1], columns=["Frecuencia por clase"])
    print(num_lab)
    print("-----------------------------------")

    # Save Frecuencia por clase to results.json
    results_path = os.path.join("app", "bnn", "output", "results.json")
    class_counts = {f"class_{int(rr[0][i])}": int(rr[1][i]) for i in range(len(rr[0]))}

    try:
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
        else:
            results = {}

        # Add or update the class frequency data for this CV fold
        if "class_frequency" not in results:
            results["class_frequency"] = {}

        results["class_frequency"][f"cv_fold_{ite+1}"] = class_counts

        # Ensure directory exists
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        # Save updated results
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

    except Exception as e:
        print(f"Error saving class frequency to results.json: {e}")

    return train_loader, X_te, Y_te, fin
