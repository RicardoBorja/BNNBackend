from __future__ import division
import numpy as np
import torch
import time as t
import pytensor.tensor as tt
import pymc as pm
from torch.autograd import Variable
from app.bnn.redneuronal_bay.utils import *
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import os
import json
from abc import abstractmethod, ABCMeta

# Get the absolute path to the project root
base_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
output_folder = os.path.join(base_dir, "app", "bnn", "output") + os.path.sep
plots_folder = os.path.join(base_dir, "app", "data", "plots") + os.path.sep
cv_plots_folder = os.path.join(plots_folder, "cv") + os.path.sep

os.makedirs(output_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)
os.makedirs(cv_plots_folder, exist_ok=True)

"Clase base para todos los optimizadores"


class BaseOptimizer:
    """
    Es una clase base y no debe ser instanciada
    """

    __metaclass__ = ABCMeta

    def __init__(self, learning_rate, decay, momentum, Bay, img, image_size):

        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.Bay = Bay
        self.img = img
        self.image_size = image_size
        self._gradients = {}

    def _forwardprop(self, Rn, x, run, trace=True):
        """
        Forward propogation metodo para la red neuronal.
        :nn: red neuronal tipo objeto
        :x: datos de entrada
        :trace: (Binario)por default es True
        :regresa: Si trace es true, la funcion devuelve lla entrada de cada capa y la salida de la ultima capa.
        Si trace es False, La funcion devuelve la salida de la ultima capa.

        'Ejemplo: Si tenemos una entrada y dos capas ocultas la salida de forward con trace = True es' \
        '[ (input, None),(input,None),(input,output)]'
        """
        # print('ingreso',x.shape)
        if trace == True:
            outputs = []

        if self.img == True:
            enput = x.view(-1, self.image_size)
        else:
            enput = x

        for i in range(len(Rn.layersObject)):

            layer = Rn.layersObject[i]

            a = 0
            if Rn.epoch == (run + 1):
                a = 1

            Output = layer.funcion_activacion(
                torch.add(torch.matmul(enput, layer.weights), layer.bias), a
            )
            Output = Variable(torch.FloatTensor(Output), requires_grad=True)
            if trace == True:
                if i == len(Rn.layersObject) - 1:
                    outputs.append((enput, Output))

                else:
                    outputs.append((enput, None))
            enput = Output
            # print(Output)

        if trace == False:
            outputs = Output

        return outputs

    def backprop(self, xy, Rn, save_mod, verbose, cv, k):

        name_model = save_mod

        # Para que trabaje si tengo cuda osea GPU
        # -----------------------------------------------------
        CUDA = False
        # CUDA = torch.cuda.is_available()

        if CUDA == True:
            self.Rn.cuda()
        # ------------------------------------------------------

        timein = t.time()

        """
        Algoritmo de Backpropogation 
        :nn: Red neuronal tipo  object
        :x: imagenes transformadas de entrenamiento
        :y: targets de entrenamiento
        :regresa: None
        """
        # np.random.seed(Rn.random_seed)
        np.random.seed()

        "La clase m agilitara determinar el accuracy y loss en el caso batch size especialmente"

        class RunningMetric:  # Esta funcion me permite que se vaya obteniendo valores promedio del traajo de la red,

            def __init__(self):
                self.S = 0
                self.N = 0

            def update(self, val, size):
                self.S += val
                self.N += size

            def __call__(self):
                return self.S / float(
                    self.N
                )  # float para tenre un resultado flotante y no una division entera

        "Aqui creo el diccionario que me almacenara los gradientes para ocuparlos en la actualizacion"
        for i in range(len(Rn.layersObject)):
            self._gradients[i] = 0

        self.set_smooth_gradients(Rn.layersObject)

        "----------------------------------------------------------"
        "Genera lotes de datos dependiendo del tamaño del lote."
        "----------------------------------------------------------"

        loss_b = np.array([])  # guarada la perdida en cada iteración
        loss_b2 = np.array([])
        ac_b = np.array([])
        ac_b2 = np.array([])

        run = 0
        while run < Rn.epoch:

            self.decay_learning_rate(run)

            # optimizer.zero_grad()

            ite_act = 0
            # iteraciones = 0
            num_batches = len(xy)
            running_loss = RunningMetric()  # Perdida
            running_acc = RunningMetric()  # Accuracy

            for inputs, targets in xy:
                x, y = inputs, targets

                # Para el caso CUDA ------------------------------------------
                if CUDA == True:
                    x = x.cuda()
                    y = y.cuda()
                # ------------------------------------------------------------

                n_total_row = len(y)
                # print(xtrain.shape)
                # print(ytrain.shape)

                each_layer_output = self._forwardprop(Rn, x, run)

                reversedlayers = list(range(len(Rn.layersObject))[::-1])
                # print(reversedlayers)
                outputlayerindex = reversedlayers[0]
                imgcurr = self.img  # Nuevo

                for i in reversedlayers:

                    if i == outputlayerindex:

                        actbay = 1  # Nuevo

                        ent = Variable(each_layer_output[i][1], requires_grad=True)

                        layerout, targ = ent, y

                        F_sigmoide = 0
                        if len(layerout[1]) == 1:
                            pred = torch.round(layerout)
                            F_sigmoide = 1
                        else:
                            _, pred = torch.max(layerout, 1)

                        ac1 = torch.sum(pred == targ)
                        ac = torch.sum(pred == targ).float() / n_total_row
                        ac_b = np.append(ac_b, ac)  # Para graficar funcion de accuracy

                        Losgr = Rn.criteria(targ, layerout)

                        Losgr.backward()
                        loss_b = np.append(
                            loss_b, (Losgr.data / num_batches)
                        )  # Para graficar funcion de perdida

                        running_loss.update(Losgr * n_total_row, n_total_row)
                        running_acc.update(
                            ac1.float(), n_total_row
                        )  # Actualizacion y casteo a flotante

                        gradien = ent.grad.data
                        gradient_out = gradien
                        delta = self.calculate_delta(gradient_out, Losgr)

                    else:
                        if imgcurr == True:
                            actbay = 0
                        else:
                            actbay = 1

                        delta = self.calculate_delta(
                            Rn.layersObject[i].derivative(each_layer_output[i + 1][0]),
                            torch.matmul(delta, Rn.layersObject[i + 1].weights.T),
                        )

                    Rn.layersObject[i].weights = self.update_weights(
                        delta,
                        layer_input=each_layer_output[i][0],
                        layer_index=i,
                        curr_weights=Rn.layersObject[i].weights,
                        Lambda=Rn.Lambda,
                        Bay=self.Bay,
                        img=self.img,
                        image_size=self.image_size,
                        ite_act=ite_act,
                        total=num_batches,
                        acbay=actbay,
                    )

                    Rn.layersObject[i].bias = self.update_bias(
                        delta,
                        curr_bias=Rn.layersObject[i].bias,
                        Bay=self.Bay,
                        img=self.img,
                        ite_act=ite_act,
                        total=num_batches,
                        layer_index=i,
                        acbay=actbay,
                    )

                ite_act += 1
                if Rn.verbose == True:
                    progreso_epo(
                        ite_act=ite_act,
                        total=num_batches,
                        total_epoch=Rn.epoch,
                        act_epoch=run + 1,
                        prefix="Progress:",
                    )

            "Calcula la perdida de entrenamiento loss(SSE)"
            loss_b2 = np.append(loss_b2, running_loss().detach().numpy())
            ac_b2 = np.append(ac_b2, running_acc().detach().numpy())

            loss = np.round(Losgr.data.item(), 4)
            k_nam = str(k)

            if cv == True:
                name2 = "best_" + name_model + "_K" + k_nam
            else:
                name2 = "best_" + name_model

            if run == 0:
                best_model = Rn.layersObject
                acc_ini = running_acc().data
                best_acc = acc_ini  # Initialize best_acc with initial accuracy
                ep = 0  # Also initialize ep to avoid similar issues
            else:
                if running_acc().data >= acc_ini:
                    ep = run
                    best_model = Rn.layersObject
                    torch.save(best_model, output_folder + name2)
                    best_acc = running_acc().data
                    acc_ini = best_acc

            ru_los = np.asarray(running_loss().data)
            ru_acc = np.asarray(running_acc().data)

            if Rn.verbose == True:

                print(
                    f"learning rate:{self.learning_rate} loss:{np.round(ru_los*1.0,4)}"
                )

                if F_sigmoide == 1:
                    print(f"Accuracy: {np.round(ru_acc*1,2)}%")
                else:
                    print(f"Accuracy: {np.round(ru_acc*100.0,2)}%")

            run += 1

        timeout = t.time()

        # Para hacer las graficas de Loss y accuracy

        # ---------------------------------------------------------------------------------
        # Para luego graficar el accuracy normal versus el bayesiano grabo los valores
        Bay = self.Bay
        if Bay == False:
            if cv == True:
                n1 = "Acc" + name_model + "_K" + k_nam
                n2 = "Loss" + name_model + "_K" + k_nam
            else:
                n1 = "Acc"
                n2 = "Loss"

        else:
            if cv == True:
                n1 = "Acc_Bay" + name_model + "_K" + k_nam
                n2 = "Loss_Bay" + name_model + "_K" + k_nam
            else:
                n1 = "Acc_Bay"
                n2 = "Loss_Bay"

        # ojo esto solo me sirve para las pruebas para yo poder graficar, guardo los acc y lost
        torch.save(ac_b2, output_folder + n1)
        torch.save(loss_b2, output_folder + n2)
        torch.save(run, output_folder + "epoch")  # Se podria quitar ojo revisar bien
        # -------------------------------------------------------------------------------
        # Para grafica, si no hay cv hace todas las graficas caso contrario no

        if cv == True:
            print("")

        else:

            # Grafica de Loss tomando en cuenta los batch
            ### cambio por actualizacion de la nueva libreria ###
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(10, 7))
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("With batch: Loss - Train")
            plt.plot(np.arange(len(loss_b)), loss_b)
            # Save plot instead of displaying it
            plt.savefig(f"{plots_folder}loss_with_batch.png")
            plt.close()

            # Grafica de Loss sin batch solo con epochs
            plt.figure(figsize=(10, 7))
            sns.set_theme(style="whitegrid")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Loss - Train")
            plt.plot(np.arange(1, Rn.epoch + 1), loss_b2)
            # Save plot instead of displaying it
            plt.savefig(f"{plots_folder}loss_by_epoch.png")
            plt.close()

            # Grafica de Accuracy tomando en cuenta los batch
            plt.figure(figsize=(10, 7))
            sns.set_theme(style="whitegrid")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title("With batch: Accuracy - Train")
            plt.plot(np.arange(len(ac_b)), ac_b)
            # Save plot instead of displaying it
            plt.savefig(f"{plots_folder}accuracy_with_batch.png")
            plt.close()

            # Grafica de Accuracy sin batch solo con epochs
            plt.figure(figsize=(10, 7))
            sns.set_theme(style="whitegrid")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title("Accuracy - Train")
            plt.plot(np.arange(1, Rn.epoch + 1), ac_b2)
            # Save plot instead of displaying it
            plt.savefig(f"{plots_folder}accuracy_by_epoch.png")
            plt.close()

        # return (loss_b, ac_b,Rn) # Revisar porque no me retorna los datos
        # print(predicted)
        # print(np.round(predicted,3))

        # Tiempo de ejecución
        print("---------------------------------------")
        print("The process took = ", round((timeout - timein) / 60, 4), "minutes")
        print("---------------------------------------")

        # Guardar el modelo

        print("---------------------------------------")
        torch.save(Rn.layersObject, output_folder + name_model)
        print("Final model saved as:", name_model)
        if F_sigmoide == 1:
            print(f"Final accuracy = {np.round(ru_acc*1,2)}%")
        else:
            print(f"Final accuracy = {np.round(ru_acc*100.0,2)}%")
        print("---------------------------------------")

        # name2 = 'best_'+name_model
        print("---------------------------------------")
        # torch.save(best_model, name2)
        print("Best model saved as:", name2)
        best_ru_acc = np.asarray(best_acc)
        if F_sigmoide == 1:
            print(f"Best accuracy = {np.round(best_ru_acc*1,2)}%")
        else:
            print(f"Best accuracy = {np.round(best_ru_acc*100.0,2)}% in epoch:{ep}")

        # save the accuracy value into the bnn/output/results.json file in the accuracy variable
        try:
            # Read existing results if file exists
            if os.path.exists(output_folder + "results.json"):
                with open(output_folder + "results.json", "r") as f:
                    results = json.load(f)
            else:
                results = {}

            # Update accuracy and epoch without overwriting other data
            results["accuracy"] = float(np.round(best_ru_acc * 100.0, 2))
            results["epoch"] = int(ep)

            # Write the updated results back to the file
            with open(output_folder + "results.json", "w") as f:
                json.dump(results, f, indent=4)

        except Exception as e:
            print(f"Error saving accuracy to results.json: {e}")

        print("---------------------------------------")
        print("-------Entrenamiento terminado---------")
        print("---------------------------------------")

        for i in range(len(Rn.layersObject)):
            self._gradients[i] = 0

        salid = (
            Rn.layersObject
        )  # Se realiza esto para que guarde el modelo que tendra como salida y aplicar la funcion para
        #                          que los pesos regresen a un valor aleatorio (esto es para CV) - en utils act_Weig_bias
        act_Weig_bias(Rn)
        salida = act_Weig_bias(Rn)

        return salida, ac_b2, loss_b2

    @abstractmethod
    def update_weights(
        self, delta, layer_input, layer_index, curr_weights, Lambda, Bay, rseed
    ):
        """
        Al ser una funcion abstracta se divide para cada clase derivada
        :delta:
        :layer_input: entrada de cada capa
        :layer_index:Indices de capas a ser actualizadas
        :curr_weights: Capa actual de pesos
        :Lambda: Parámetro de regularizacion.
        :regreso: Pesos actualizados
        """
        pass

    def update_bias(
        self, delta, curr_bias, Bay, img, ite_act, total, layer_index, acbay
    ):
        """
        Al ser una funcion abstracta se divide para cada clase derivada
        :delta:
        :curr_bias: current bias of layer.
        :regreso: Sesgos actualizados
        """
        # --------Proceso bayesiano-------------------------------------------------------
        img = self.img

        # if Bay==True: # Para que haga el bayesiano en cada batch de cada epoch
        if (
            Bay == True and ite_act == (total - 1) and acbay == 1
        ):  # Para que haga el bayesiano en el ultimo batch de cada epoch

            try:
                from app.config import conf_manager

                # Get bayesian_config directly from nn_parameters.conf
                nn_params = conf_manager.get_value("nn_parameters")

                # Check if bayesian_config exists in nn_params
                if nn_params and "bayesian_config" in nn_params:
                    bayesian_config = nn_params["bayesian_config"]
                else:
                    # Fall back to standard configuration
                    bayesian_config = {
                        "distribution_type": "normal",
                        "mean": 0.0,
                        "sigma": 1.0,
                        "alpha": 0.1,
                        "beta": 0.1,
                        "lambdaPar": 0.1,
                    }

                # Extract parameters with defaults
                distribution_type = bayesian_config.get("distribution_type", "normal")
                prior_mean = bayesian_config.get("mean", 0.0)
                prior_sigma = bayesian_config.get("sigma", 1.0)
                prior_alpha = bayesian_config.get("bayesian_alpha", 0.1)
                prior_beta = bayesian_config.get("beta", 0.1)
                prior_lambda = bayesian_config.get("lambdaPar", 1.0)

                # --------------------------------------------
                # Para que no presente en consola el proceso
                logger = logging.getLogger("pymc")
                logger.setLevel(logging.CRITICAL)
                # --------------------------------------------

                # Analisis Bayesiano----------------------------------------------------------
                delta_numpy = delta.data.numpy()

                # Para hacer que el tunning varie por capas mas numerosas
                tunb = 10

                # Compute the summed_delta outside of PyMC model
                summed_delta_numpy = np.sum(delta_numpy, axis=0, keepdims=True)

                # Clear PyMC cache between runs
                with pm.Model() as model:
                    # Generate a simple model ID
                    model_id = f"b{layer_index}_{np.random.randint(10000)}"

                    # Choose prior distribution based on distribution_type
                    if distribution_type == "normal":
                        delt_b = pm.Normal(
                            f"db_{model_id}",
                            mu=prior_mean + delta_numpy,
                            sigma=prior_sigma,
                            shape=delta_numpy.shape,
                        )
                    elif distribution_type == "halfnormal":
                        # HalfNormal has no location parameter, only scale
                        delt_b = pm.HalfNormal(
                            f"db_{model_id}",
                            sigma=prior_sigma,
                            shape=delta_numpy.shape,
                        )
                        # We add the delta_numpy after sampling
                        delt_b = delt_b + delta_numpy + prior_mean
                    elif distribution_type == "exponential":
                        # Exponential has no location parameter
                        delt_b = pm.Exponential(
                            f"db_{model_id}",
                            lam=1
                            / prior_lambda,  # Convert sigma to lambda rate parameter
                            shape=delta_numpy.shape,
                        )
                        # We add the delta_numpy and mean after sampling
                        delt_b = delt_b + delta_numpy + prior_mean
                    elif distribution_type == "cauchy":
                        delt_b = pm.Cauchy(
                            f"db_{model_id}",
                            alpha=prior_alpha + delta_numpy,  # Location
                            beta=prior_beta,  # Scale
                            shape=delta_numpy.shape,
                        )
                    else:
                        # Default to normal if unknown
                        delt_b = pm.Normal(
                            f"db_{model_id}",
                            mu=delta_numpy,
                            sigma=0.01,
                            shape=delta_numpy.shape,
                        )

                    # Standard deviation for observation noise
                    sd_b = pm.HalfNormal(f"sd_{model_id}", sigma=10.0)

                    # Use PyTensor's sum with the correct syntax
                    delta_b = pm.Deterministic(
                        f"delta_{model_id}", tt.sum(delt_b, axis=0)
                    )

                    # Define likelihood using precomputed summed_delta
                    obs_pos_b = pm.Normal(
                        f"obs_{model_id}",
                        mu=delta_b,
                        sigma=sd_b,
                        observed=summed_delta_numpy,
                    )

                    # Elijo el modelo y obtengo muestras
                    step = pm.NUTS()
                    trace = pm.sample(
                        5,
                        step=step,
                        tune=tunb,
                        cores=1,
                        chains=1,
                        compute_convergence_checks=False,
                        progressbar=True,
                    )

                    # Extract the mean directly inside the context manager
                    db_values = trace[f"delta_{model_id}"]
                    db_mean = db_values.mean(axis=0)

                # Convert back to PyTorch Variable outside the context
                db = Variable(torch.FloatTensor(db_mean), requires_grad=True)
                gradient = db

            except Exception as e:
                # Fallback to standard gradient if Bayesian update fails
                print(f"Bayesian update failed, using standard gradient: {str(e)}")
                gradient = torch.sum(delta, dim=0, keepdim=True)
        else:
            gradient = torch.sum(delta, dim=0, keepdim=True)

        # Update bias with gradient
        new_bias = curr_bias - self.learning_rate * gradient
        return new_bias

    def calculate_delta(self, derivative, loss):
        """
        Calcula los valores de delta
        :derivative: deriva la capa.
        :param loss: loss de cada capa.
        :regresa: delta
        """
        # print('loss')
        # print(loss)
        # print('derivada')
        # print(derivative)
        # epss = 1e-10
        return torch.mul(derivative, loss)
        # return (torch.mul(derivative, loss)+epss)

    def decay_learning_rate(self, run):
        """
        Perite que la tasa de aprendizaje decaiga dependiendo
        del numero de epochs
        :run:
        :regreso:
        """
        pass

    def set_smooth_gradients(self, Rn):
        """
        Solo se usa en Adam.
        :nn: red neuronal
        :regreso: None
        """
        pass
