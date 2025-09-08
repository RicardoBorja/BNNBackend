from __future__ import division
from app.bnn.redneuronal_bay.Optimizers.base_optimizer import BaseOptimizer
import numpy as np
import copy
import pytensor.tensor as tt
import torch
from torch.autograd import Variable
import pymc as pm
import warnings
import time as t
import logging

warnings.filterwarnings("ignore")


class SGD(BaseOptimizer):
    """
    Gradiente estocástico descendiente.
    """

    def __init__(self, learning_rate, decay, momentum, Bay, img, image_size):
        BaseOptimizer.__init__(
            self, learning_rate, decay, momentum, Bay, img, image_size
        )

    def update_weights(
        self,
        delta,
        layer_input,
        layer_index,
        curr_weights,
        Lambda,
        Bay,
        img,
        image_size,
        ite_act,
        total,
        acbay,
    ):
        # np.random.seed(rseed)
        img = self.img
        # np.random.seed()

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
                prior_alpha = bayesian_config.get("bayesian_alpha", 0.1)  # Use correct parameter name
                prior_beta = bayesian_config.get("beta", 0.1)
                prior_lambda = bayesian_config.get("lambdaPar", 1.0)

                # --------------------------------------------
                # Para que no presente en consola el proceso
                logger = logging.getLogger("pymc")
                logger.setLevel(logging.CRITICAL)
                # --------------------------------------------

                # Analisis Bayesiano----------------------------------------------------------
                delta_numpy = delta.data.numpy()
                layer_input_numpy = layer_input.detach().numpy()
                layer_input_numpy = layer_input_numpy.T

                # Para hacer que el tunning varie por capas mas numerosas
                tun = 5

                # Compute the actual dot product to use as observed data
                observed_dot_product = np.dot(layer_input_numpy, delta_numpy)

                with pm.Model() as NNB:
                    # Choose prior distribution based on distribution_type
                    if distribution_type == "normal":
                        delt = pm.Normal(
                            "delt",
                            mu=prior_mean + delta_numpy,
                            sigma=prior_sigma,
                            shape=delta_numpy.shape,
                        )
                    elif distribution_type == "halfnormal":
                        delt = pm.HalfNormal(
                            "delt",
                            sigma=prior_sigma,
                            shape=delta_numpy.shape,
                        )
                        # Add delta_numpy after sampling
                        delt = delt + delta_numpy + prior_mean
                    elif distribution_type == "exponential":
                        delt = pm.Exponential(
                            "delt",
                            lam=1 / prior_lambda,
                            shape=delta_numpy.shape,
                        )
                        # Add delta_numpy after sampling
                        delt = delt + delta_numpy + prior_mean
                    elif distribution_type == "cauchy":
                        delt = pm.Cauchy(
                            "delt",
                            alpha=prior_alpha + delta_numpy,  # Location parameter
                            beta=prior_beta,  # Scale parameter
                            shape=delta_numpy.shape,
                        )
                    else:
                        # Default to normal if distribution type is not recognized
                        delt = pm.Normal(
                            "delt", mu=delta_numpy, sigma=0.01, shape=delta_numpy.shape
                        )

                    # Define standard deviation for noise
                    sd = pm.HalfNormal("sd", sigma=10.0)

                    # Define the delta_w as a deterministic variable
                    delta_w = pm.Deterministic(
                        "delta_w", tt.dot(layer_input_numpy, delt)
                    )

                    # Use observed_dot_product as the observed value instead of delta_w
                    obs_pos = pm.Normal(
                        "obs_pos", mu=delta_w, sigma=sd, observed=observed_dot_product
                    )

                    # Choose the sampling method
                    step = pm.NUTS()

                    # Obtain posterior samples
                    trace = pm.sample(
                        5,
                        step=step,
                        tune=tun,
                        cores=1,
                        chains=1,
                        compute_convergence_checks=False,
                        progressbar=True,
                    )

                # Extract the posterior mean for delta_w
                dw = trace["delta_w"]
                dw = dw.mean(axis=0)

                # Convert back to PyTorch Variable
                dw = Variable(torch.FloatTensor(dw), requires_grad=True)
                gradient = dw

            except Exception as e:
                print(f"Bayesian update failed, using standard gradient: {str(e)}")
                gradient = torch.matmul(layer_input.T, delta)
        else:
            gradient = torch.matmul(layer_input.T, delta)

        # Rest of the update method remains unchanged
        eps = 0
        self._gradients[layer_index] = torch.mul(
            self.momentum, self._gradients[layer_index]
        ) - torch.mul(self.learning_rate, gradient)

        return curr_weights + self._gradients[layer_index] + eps

    # def update_weights(
    #     self,
    #     delta,
    #     layer_input,
    #     layer_index,
    #     curr_weights,
    #     Lambda,
    #     Bay,
    #     img,
    #     image_size,
    #     ite_act,
    #     total,
    #     acbay,
    # ):

    #     # np.random.seed(rseed)
    #     img = self.img
    #     # np.random.seed()

    #     # if Bay==True: # Para que haga el bayesiano en cada batch de cada epoch
    #     if (
    #         Bay == True and ite_act == (total - 1) and acbay == 1
    #     ):  # Para que haga el bayesiano en el ultimo batch de cada epoch

    #         # --------------------------------------------
    #         # Para que no presente en consola el proceso
    #         logger = logging.getLogger("pymc3")
    #         # logger.propagate = False
    #         logger.setLevel(logging.CRITICAL)
    #         # --------------------------------------------

    #         # Analisis Bayesiano----------------------------------------------------------
    #         # delta = np.asarray(delta)
    #         # print(delt.dtype)

    #         # print(observ.shape)
    #         delta = delta.data.numpy()
    #         # print(delta)
    #         learning_rate = self.learning_rate
    #         # layer_input = np.asarray(layer_input.T)
    #         # print(layer_input)
    #         # print(layer_input.dtype)
    #         layer_input = layer_input.detach().numpy()
    #         # print('numpy')
    #         # print(layer_input)
    #         layer_input = layer_input.T
    #         # print('X es')
    #         # print(layer_input)
    #         # print(layer_input.shape)
    #         # delt_w = np.dot(layer_input,delta)
    #         # print('sin baye')
    #         # print(delt_w)

    #         # Para hacer que el tunning varie por capas mas numerosas
    #         mues = layer_index  # es el numero de capa
    #         tun = 5
    #         # print(mues)
    #         # ------------parañlizado al momento ---- por acbay
    #         # if img==True:
    #         #    if mues==1:
    #         #        tun = 200
    #         # print('tuning de:',tun)
    #         #    elif mues==0:
    #         #        tun= 100
    #         # print('tuning de:',tun)
    #         #    else:
    #         #        tun = 300
    #         # print('tuning de:',tun)
    #         # -----------------------------------------------
    #         # print('transpuesta')
    #         # print(layer_input)

    #         # timein = t.time()

    #         with pm.Model() as NNB:
    #             # Defino priors
    #             delt = pm.Normal("delt", mu=delta, sigma=0.01, shape=delta.shape)
    #             # print('modelo bay_ delt')
    #             # print(delt.shape)
    #             # sd = pm.HalfNormal('sd', sigma=1)
    #             sd = pm.Uniform("sd", 0, 100)

    #             # Defino valores en funcion de las priors

    #             delta_w = pm.Deterministic("delta_w", ((tt.dot(layer_input, delt))))
    #             # --delta_w = tt.dot(layer_input, delta)

    #             # Defino likelihood en funcion de los valores establecidos y priors
    #             # obs_pos = pm.Normal('obs_pos', mu=delta_w, sigma=sd,observed=True)
    #             obs_pos = pm.Normal("obs_pos", mu=delta_w, sigma=sd, observed=delta_w)
    #             # obs_pos = pm.Normal('obs_pos', mu=delta_w, sigma=sd,observed=False)

    #             # Elijo el modelo
    #             # --start = pm.find_MAP() # esta opcion permite que encuentre los valores iniciales para optimizacion
    #             # step = pm.NUTS(target_accept=.95)
    #             # step = pm.HamiltonianMC()
    #             # --step = pm.NUTS(state=start)
    #             step = pm.NUTS()
    #             # Obtengo las muestras posteriores
    #             # trace = pm.sample(500, step=step, tune=2500, cores=1, chains=1, compute_convergence_checks=True, progressbar=True,  random_seed=42)
    #             # trace = pm.sample(500, step=step, tune=400, cores=2, chains=2, compute_convergence_checks=False, progressbar=True)
    #             trace = pm.sample(
    #                 5,
    #                 step=step,
    #                 tune=tun,
    #                 cores=4,
    #                 chains=1,
    #                 compute_convergence_checks=False,
    #                 progressbar=True,
    #             )

    #         # timeout = t.time()

    #         dw = trace["delta_w"]
    #         dw = dw.mean(axis=0)
    #         # print('Modificación bayesiana')
    #         # print(dw)
    #         # print('El proceso tardó = ', round((timeout-timein)/60,4),'minutos')
    #         # print(camelia)
    #         # -------------------------------------------------------------------------------
    #         # Para hacer pruebas de modelo bayesiano y ver que converja - gráficas
    #         # print(pm.summary(trace))
    #         # print(sd)
    #         # print(pm.plot_posterior(trace['delta_w']))
    #         # print(pm.traceplot(trace))
    #         # print(pm.forestplot(trace, r_hat=True))
    #         # print(miraar)
    #         # -------------------------------------------------------------------------------

    #         dw = Variable(torch.FloatTensor(dw), requires_grad=True)
    #         gradient = dw  # - Lambda * curr_weights
    #         # ------------------------------------------------------------------------------

    #     else:
    #         gradient = torch.matmul(layer_input.T, delta)  # - Lambda * curr_weights

    #     # print(layer_input)
    #     # print('transpuesta')
    #     # print(layer_input.T)
    #     # print(layer_input.T.shape)
    #     # print(delta)
    #     # print(delta.shape)
    #     # print(self.momentum)
    #     # print(self._gradients[layer_index])
    #     # print(self._gradients[layer_index].shape)
    #     # print(self.learning_rate)
    #     # print(self.learning_rate.shape)
    #     # print(gradient)
    #     # print('pesos actuales')
    #     # print(curr_weights)
    #     # print(gradient.shape)
    #     # eps = 1e-8
    #     eps = 0
    #     self._gradients[layer_index] = torch.mul(
    #         self.momentum, self._gradients[layer_index]
    #     ) - torch.mul(self.learning_rate, gradient)

    #     # ------------------------------------------------------------------------------------------
    #     # fug_grad= self._gradients[layer_index] # para contrarestar en casos de fuga de gradiente
    #     # borde_incre = 0.5/total  # generalmente 0.5
    #     # fug_grad[fug_grad>=borde_incre] = borde_incre
    #     # fug_grad[fug_grad<=-borde_incre] = -borde_incre
    #     # self._gradients[layer_index] = fug_grad
    #     # -------------------------------------------------------------------------------------------

    #     return curr_weights + self._gradients[layer_index] + eps

    def decay_learning_rate(self, run):
        """
        Para que decaiga la tasa de aprendizaje segun el numero de epochs.
        :run:
        :regreso:
        """
        self.learning_rate *= 1 / (1 + self.decay * run)


class RMSProp(BaseOptimizer):

    def __init__(self, learning_rate, decay, momentum, Bay, img, image_size):
        BaseOptimizer.__init__(
            self, learning_rate, decay, momentum, Bay, img, image_size
        )

    def update_weights(
        self,
        delta,
        layer_input,
        layer_index,
        curr_weights,
        Lambda,
        Bay,
        img,
        image_size,
        ite_act,
        total,
        acbay,
    ):
        # Use PyTorch operations instead of NumPy
        gradient = torch.matmul(layer_input.T, delta)
        if Lambda > 0:
            gradient = gradient - Lambda * curr_weights

        eps = 1e-8

        self._gradients[layer_index] = self.decay * self._gradients[layer_index] + (
            1 - self.decay
        ) * torch.pow(gradient, 2)

        return curr_weights - self.learning_rate * gradient / (
            torch.sqrt(self._gradients[layer_index]) + eps
        )


class Adam(BaseOptimizer):

    def __init__(self, learning_rate, decay, momentum, Bay, img, image_size):
        BaseOptimizer.__init__(
            self, learning_rate, decay, momentum, Bay, img, image_size
        )
        self.beta1 = 0.9
        self.beta2 = 0.99
        self._smooth_gradients = {}

    def set_smooth_gradients(self, nn):
        # If nn is already a defaultdict (this is what's happening in your case)
        if isinstance(nn, dict) or hasattr(nn, "items"):
            for i in range(len(nn)):
                self._smooth_gradients[i] = 0
        # If nn is a RedNeuBay instance with a layersObject attribute
        elif hasattr(nn, "layersObject"):
            for i in range(len(nn.layersObject)):
                self._smooth_gradients[i] = 0
        # For any other iterable
        elif hasattr(nn, "__len__"):
            for i in range(len(nn)):
                self._smooth_gradients[i] = 0
        else:
            raise TypeError(
                "Expected nn to be iterable or have a layersObject attribute"
            )

    def update_weights(
        self,
        delta,
        layer_input,
        layer_index,
        curr_weights,
        Lambda,
        Bay,
        img,
        image_size,
        ite_act,
        total,
        acbay,
    ):
        gradient = torch.matmul(layer_input.T, delta)  # - Lambda * curr_weights

        eps = 1e-8

        self._smooth_gradients[layer_index] = torch.mul(
            self.beta1, self._smooth_gradients[layer_index]
        ) + torch.mul((1 - self.beta1), gradient)

        self._gradients[layer_index] = torch.mul(
            self.beta2, self._gradients[layer_index]
        ) + torch.mul((1 - self.beta2), torch.pow(gradient, 2))

        return curr_weights - torch.mul(
            self.learning_rate, self._smooth_gradients[layer_index]
        ) / (torch.sqrt(self._gradients[layer_index]) + eps)


class Nesterov(BaseOptimizer):

    def __init__(self, learning_rate, decay, momentum, Bay, img, image_size):
        BaseOptimizer.__init__(
            self, learning_rate, decay, momentum, Bay, img, image_size
        )

    def update_weights(
        self,
        delta,
        layer_input,
        layer_index,
        curr_weights,
        Lambda,
        Bay,
        img,
        image_size,
        ite_act,
        total,
        acbay,
    ):

        gradient = torch.matmul(layer_input.T, delta)  # - Lambda * curr_weights

        # This line has the error - copy.copy is a function, not an object with a 'self' attribute
        prev_velocity = copy.copy(
            self._gradients[layer_index]
        )  # Fixed: changed from copy.copy.self

        new_velocity = torch.mul(
            self.momentum, self._gradients[layer_index]
        ) - torch.mul(self.learning_rate, gradient)

        # Store the new velocity for next iteration
        self._gradients[layer_index] = new_velocity

        return (
            curr_weights
            - torch.mul(self.momentum, prev_velocity)
            + torch.mul((1 + self.momentum), new_velocity)
        )

    def decay_learning_rate(self, run):
        """
        This function decays the learning rate depending
        on the number of runs and decay rate.
        :param run:
        :return:
        """
        self.learning_rate *= 1 / (1 + self.decay * run)


class Adagrad(BaseOptimizer):

    def __init__(self, learning_rate, decay, momentum, Bay, img, image_size):
        BaseOptimizer.__init__(
            self, learning_rate, decay, momentum, Bay, img, image_size
        )

    def update_weights(
        self,
        delta,
        layer_input,
        layer_index,
        curr_weights,
        Lambda,
        Bay,
        img,
        image_size,
        ite_act,
        total,
        acbay,
    ):

        eps = 1e-8

        gradient = torch.matmul(layer_input.T, delta)  # - Lambda * curr_weights

        self._gradients[layer_index] = torch.pow(gradient, 2)

        return curr_weights - torch.mul(self.learning_rate, gradient) / (
            torch.sqrt(self._gradients[layer_index]) + eps
        )
