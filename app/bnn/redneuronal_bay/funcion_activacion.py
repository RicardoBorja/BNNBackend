from __future__ import division
import numpy as np
import torch.nn.functional as F
import pymc as pm
import pytensor.tensor as tt
import torch
from xarray import Variable
import json
import os
import re


"""
Defino las funciones de activacion.
"""

# Semilla aleatoria para mayor solidez en la funcion noise relu.
np.random.seed(1990)


def sigmoide(tensor, a):
    """
    Funcion Sigmoide.
    :Entrada= Debe ser un "numpy array"
    """
    return F.sigmoid(tensor)


def tan_h(tensor, a):
    """
    Función tangente hiperbolica
    :Entrada= Debe ser un "numpy array"

    """
    return F.tanh(tensor)


def ReLU(tensor, a):
    """
    Funcion Rectified Linear Unit
    :Entrada = Debe ser un "numpy array"

    """
    return F.relu(tensor)


def Noisy_ReLU(tensor, a):
    """
    Nosiy Rectified Linear Unit. T.
    :tensor: numpy array
    :regresa: numpy array de elementos rectificados.
    """
    return np.maximum(0.0, tensor + np.random.normal(0.0, 1.0))


def Leaky_ReLU(tensor, a):
    """
    Leaky Rectified Linear Unit.
    :tensor: numpy array
    :regresa: numpy array .
    """
    return F.leaky_relu(tensor)


# ...existing code...


def parse_vector_param(param, size=None, default=1.0):
    """
    Converts a comma-separated string or list to a numpy array of floats.
    If param is None, returns a vector of 'default' of length 'size'.
    """
    if param is None or (isinstance(param, str) and param.strip() == ""):
        if size is not None:
            return np.ones(size) * default
        return np.array([default])

    if isinstance(param, str):
        try:
            arr = np.array(
                [float(x.strip()) for x in param.split(",") if x.strip() != ""]
            )
            if size is not None and arr.size != size:
                raise ValueError(
                    f"Vector length {arr.size} does not match expected size {size}"
                )
            return arr
        except Exception as e:
            print(f"Error parsing vector parameter '{param}': {e}")
            if size is not None:
                return np.ones(size) * default
            return np.array([default])
    elif isinstance(param, (list, np.ndarray)):
        arr = np.array(param, dtype=float)
        if size is not None and arr.size != size:
            raise ValueError(
                f"Vector length {arr.size} does not match expected size {size}"
            )
        return arr
    else:
        if size is not None:
            return np.ones(size) * default
        return np.array([default])


def softmax(tensor, a):
    """
    Funcion Softmax, similara a la funcion exponencial normalizada: Entrega un vector de probabilides de acierto que sumados dan 1.
    :Entrada: Valores en 1D y del tipo numpy array.

    """
    return F.softmax(tensor, dim=1)


def softmax_Bay(tensor, a):
    """
    Funcion Softmax bayesiana-debe señalar el vector de probabilidades a prior: Entrega un vector de probabilides de acierto,sumados dan 1.
    :Entrada: Valores en 1D y del tipo numpy array.

    """
    if a == 0:
        # print(tensor)
        observed_soft = F.softmax(tensor, dim=1)
        observed_soft = observed_soft.data.numpy()
        tensor = tensor.data.numpy()

        # Fix: Make sure both arrays have the same number of rows
        min_rows = min(tensor.shape[0], observed_soft.shape[0])
        tensor = tensor[:min_rows]
        observed_soft = observed_soft[:min_rows]

        assert (
            tensor.shape[0] == observed_soft.shape[0]
        ), f"Shape mismatch: tensor {tensor.shape}, observed_soft {observed_soft.shape}"

        row_obser, col_obser = observed_soft.shape

        # Fix: Create prior probabilities dynamically based on actual output dimensions
        # Use uniform distribution as default prior
        prob = np.ones((1, col_obser)) / col_obser
        prob = np.repeat(prob, row_obser, axis=0)

        # Validate shapes before creating PyMC model
        assert prob.shape == (
            row_obser,
            col_obser,
        ), f"Prior shape mismatch: expected {(row_obser, col_obser)}, got {prob.shape}"
        assert tensor.shape == (
            row_obser,
            col_obser,
        ), f"Tensor shape mismatch: expected {(row_obser, col_obser)}, got {tensor.shape}"

        # Additional safeguards for numerical stability
        # Check for NaN or inf in tensor
        if np.any(np.isnan(tensor)) or np.any(np.isinf(tensor)):
            print("Warning: Tensor contains NaN or inf values, using regular softmax")
            return F.softmax(torch.tensor(tensor, dtype=torch.float32), dim=1)

        # Check for extreme values that might cause numerical issues
        if np.any(np.abs(tensor) > 100):
            print(
                "Warning: Tensor contains extreme values, clipping to prevent numerical issues"
            )
            tensor = np.clip(tensor, -100, 100)

        # Get bayesian configuration directly from file
        try:
            # Get the absolute path to the config file
            # Use a robust approach that works when running with uvicorn
            import sys
            import os

            # Get the directory of the current file
            current_file_dir = os.path.dirname(os.path.abspath(__file__))

            # Try multiple possible paths
            possible_paths = [
                "app/config/nn_parameters.conf",  # If running from project root
                "config/nn_parameters.conf",  # If running from app directory
                os.path.join(
                    current_file_dir, "..", "..", "..", "config", "nn_parameters.conf"
                ),  # Relative to current file
                os.path.join(
                    current_file_dir,
                    "..",
                    "..",
                    "..",
                    "..",
                    "app",
                    "config",
                    "nn_parameters.conf",
                ),  # Alternative relative path
            ]

            config_file = None
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    config_file = abs_path
                    break

            if config_file is None:
                # Last resort: try to find the file by searching from the current file's directory
                search_dir = current_file_dir
                for _ in range(6):  # Go up to 6 levels
                    potential_path = os.path.join(
                        search_dir, "app", "config", "nn_parameters.conf"
                    )
                    if os.path.exists(potential_path):
                        config_file = potential_path
                        break
                    search_dir = os.path.dirname(search_dir)
                    if search_dir == os.path.dirname(search_dir):  # Reached root
                        break

            if config_file is None:
                raise FileNotFoundError(
                    "Could not find nn_parameters.conf in any expected location"
                )

            # Try to parse the JSON file with error handling
            try:
                with open(config_file, "r") as f:
                    content = f.read()
                    # Replace Python literals with JSON equivalents
                    content = re.sub(r"\bTrue\b", "true", content)
                    content = re.sub(r"\bFalse\b", "false", content)
                    content = re.sub(r"\bNone\b", "null", content)
                    # Remove trailing commas before closing braces/brackets
                    content = re.sub(r",\s*([}\]])", r"\1", content)
                    # Remove comments (if any)
                    content = re.sub(r"//.*", "", content)
                    # Remove extra whitespace
                    content = " ".join(content.split())
                    nn_params = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parsing error in config file: {e}")
                print("Using default bayesian configuration")
                nn_params = {}

            # Check if bayesian_config exists in nn_params
            if nn_params and "bayesian_config" in nn_params:
                bayesian_config = nn_params["bayesian_config"]
            else:
                # Fall back to standard configuration
                bayesian_config = {
                    "distribution_type": "normal",
                    "mean": 0.0,
                    "sigma": 1.0,
                }

            # Extract parameters with defaults
            distribution_type = bayesian_config.get("distribution_type", "normal")

            alpha_vector = parse_vector_param(
                bayesian_config.get("alpha_vector"), size=col_obser, default=1.0
            )
            p_vector = parse_vector_param(
                bayesian_config.get("p_vector"), size=col_obser, default=1.0 / col_obser
            )
            weights = parse_vector_param(
                bayesian_config.get("weights"), size=col_obser, default=1.0 / col_obser
            )
            means = parse_vector_param(
                bayesian_config.get("means"), size=col_obser, default=0.0
            )
            sigmas = parse_vector_param(
                bayesian_config.get("sigmas"), size=col_obser, default=1.0
            )

            for name, arr in [
                ("weights", weights),
                ("means", means),
                ("sigmas", sigmas),
            ]:
                if arr.shape != (col_obser,):
                    print(
                        f"Warning: {name} shape {arr.shape} does not match expected ({col_obser},), resizing."
                    )
                    arr = np.resize(arr, (col_obser,))
                # assign back to variable
                if name == "weights":
                    weights = arr
                elif name == "means":
                    means = arr
                elif name == "sigmas":
                    sigmas = arr

        except Exception as e:
            print(
                f"Warning: Could not load bayesian config from file, using defaults: {e}"
            )
            distribution_type = "normal"

        with pm.Model() as _:
            pm.Dirichlet("proba", a=prob, shape=(row_obser, col_obser))

            # Use parsed vector parameters for all relevant distributions
            if distribution_type == "normal":
                z = pm.Normal(
                    "z",
                    mu=bayesian_config.get("mu", 0) + tensor,
                    sigma=bayesian_config.get("sigma", 1),
                    shape=(row_obser, col_obser),
                )
            elif distribution_type == "halfnormal":
                z = pm.HalfNormal(
                    "z",
                    sigma=bayesian_config.get("sigma", 1),
                    shape=(row_obser, col_obser),
                )
                z = z + tensor + bayesian_config.get("mu", 0)
            elif distribution_type == "cauchy":
                z = pm.Cauchy(
                    "z",
                    alpha=bayesian_config.get("alpha", 0) + tensor,
                    beta=bayesian_config.get("beta", 1),
                    shape=(row_obser, col_obser),
                )
            elif distribution_type == "exponential":
                z = pm.Exponential(
                    "z",
                    lam=bayesian_config.get("lambdaPar", 1),
                    shape=(row_obser, col_obser),
                )
                z = z + tensor + bayesian_config.get("mu", 0)
            elif distribution_type == "beta":
                z = pm.Beta(
                    "z",
                    alpha=bayesian_config.get("alpha", 0),
                    beta=bayesian_config.get("beta", 1),
                    shape=(row_obser, col_obser),
                )
                z = z + tensor
            elif distribution_type == "chisquared":
                z = pm.ChiSquared(
                    "z", nu=bayesian_config.get("nu", 1), shape=(row_obser, col_obser)
                )
                z = z + tensor
            elif distribution_type == "exgaussian":
                z = pm.ExGaussian(
                    "z",
                    mu=bayesian_config.get("mu", 0),
                    sigma=bayesian_config.get("sigma", 1),
                    nu=bayesian_config.get("nu", 1),
                    shape=(row_obser, col_obser),
                )
                z = z + tensor
            elif distribution_type == "gamma":
                z = pm.Gamma(
                    "z",
                    alpha=bayesian_config.get("alpha", 0),
                    beta=bayesian_config.get("beta", 1),
                    shape=(row_obser, col_obser),
                )
                z = z + tensor
            elif distribution_type == "uniform":
                z = pm.Uniform(
                    "z",
                    lower=bayesian_config.get("lower", 0),
                    upper=bayesian_config.get("upper", 1),
                    shape=(row_obser, col_obser),
                )
                z = z + tensor
            elif distribution_type == "dirichlet":
                z = pm.Dirichlet("z", a=alpha_vector, shape=(row_obser, col_obser))
                z = z + tensor
            elif distribution_type == "multinomial":
                z = pm.Multinomial(
                    "z",
                    n=bayesian_config.get("n", 1),
                    p=p_vector,
                    shape=(row_obser, col_obser),
                )
                z = z + tensor
            elif distribution_type == "binomial":
                z = pm.Binomial(
                    "z",
                    n=bayesian_config.get("n", 1),
                    p=bayesian_config.get("p", 0.5),
                    shape=(row_obser, col_obser),
                )
                z = z + tensor
            elif distribution_type == "logistic":
                z = pm.Logistic(
                    "z",
                    mu=bayesian_config.get("mu", 0),
                    s=bayesian_config.get("scale", 1),
                    shape=(row_obser, col_obser),
                )
                z = z + tensor
            elif distribution_type == "lognormal":
                z = pm.LogNormal(
                    "z",
                    mu=bayesian_config.get("mu", 0),
                    sigma=bayesian_config.get("sigma", 1),
                    shape=(row_obser, col_obser),
                )
                z = z + tensor
            elif distribution_type == "weibull":
                z = pm.Weibull(
                    "z",
                    alpha=bayesian_config.get("alpha", 1),
                    beta=bayesian_config.get("beta", 1),
                    shape=(row_obser, col_obser),
                )
                z = z + tensor
            elif distribution_type == "bernoulli":
                z = pm.Bernoulli(
                    "z", p=bayesian_config.get("p", 0.5), shape=(row_obser, col_obser)
                )
                z = z + tensor
            elif distribution_type == "poisson":
                z = pm.Poisson(
                    "z", mu=bayesian_config.get("mu", 1), shape=(row_obser, col_obser)
                )
                z = z + tensor
            elif distribution_type == "dirichletmultinomial":
                z = pm.DirichletMultinomial(
                    "z",
                    n=bayesian_config.get("n", 1),
                    a=alpha_vector,
                    shape=(row_obser, col_obser),
                )
                z = z + tensor
            elif distribution_type == "betabinomial":
                z = pm.BetaBinomial(
                    "z",
                    n=bayesian_config.get("n", 1),
                    alpha=bayesian_config.get("alpha", 1),
                    beta=bayesian_config.get("beta", 1),
                    shape=(row_obser, col_obser),
                )
                z = z + tensor
            elif distribution_type == "categorical":
                z = pm.Categorical("z", p=p_vector, shape=(row_obser, col_obser))
                z = z + tensor
            elif distribution_type == "normalmixture":
                z = pm.NormalMixture(
                    "z", w=weights, mu=means, sigma=sigmas, shape=(row_obser, col_obser)
                )
                z = z + tensor
            elif distribution_type == "gaussianrandomwalk":
                z = pm.GaussianRandomWalk(
                    "z",
                    mu=bayesian_config.get("mu", 0),
                    sigma=bayesian_config.get("sigma", 1),
                    shape=(row_obser, col_obser),
                )
                z = z + tensor
            elif distribution_type == "ar1":
                z = pm.AR(
                    "z",
                    k=bayesian_config.get("k", 1),
                    tau=bayesian_config.get("tau", 1),
                    rho=bayesian_config.get("rho", 0.5),
                    shape=(row_obser, col_obser),
                )
                z = z + tensor
            else:
                print(
                    f"Warning: Unknown distribution type '{distribution_type}', using normal"
                )
                z = pm.Normal(
                    "z",
                    mu=tensor,
                    sigma=bayesian_config.get("sigma", 1),
                    shape=(row_obser, col_obser),
                )

            sof = pm.Deterministic("sof", tt.exp(z) / tt.sum(tt.exp(z), axis=1, keepdims=True))  # type: ignore
            pm.Categorical("obs_pos", p=sof, observed=np.argmax(observed_soft, axis=1))

            n_trials = np.ones(row_obser, dtype=int)
            step = pm.HamiltonianMC(target_accept=0.20)

            assert np.all(
                np.isfinite(observed_soft)
            ), "observed_soft contains NaN or inf"
            assert np.all(observed_soft >= 0), "observed_soft contains negative values"
            assert np.all(n_trials > 0), "n_trials contains non-positive values"
            assert np.all(np.isfinite(prob)), "prob contains NaN or inf"
            assert np.all(prob > 0), "prob contains non-positive values"

            if bayesian_config.get("markov"):
                pm.NUTS()
            else:
                pm.fit(n=1000, method="advi")

            try:
                trace = pm.sample(
                    300,
                    step=step,
                    tune=500,
                    cores=1,
                    chains=1,
                    compute_convergence_checks=False,
                    progressbar=True,
                )
                if "sof" in trace.posterior:  # type: ignore
                    soft_posteriori = (
                        trace.posterior["sof"].mean(dim=["chain", "draw"]).values  # type: ignore
                    )
                    soft_posteriori = torch.tensor(
                        soft_posteriori, dtype=torch.float32, requires_grad=True
                    )
                else:
                    raise ValueError(
                        "Variable 'sof' not found in trace. Check model definition."
                    )
            except Exception as e:
                print(
                    f"Bayesian sampling failed: {e}. Falling back to regular softmax."
                )
                soft_posteriori = F.softmax(
                    torch.tensor(tensor, dtype=torch.float32), dim=1
                )
    else:
        soft_posteriori = F.softmax(tensor, dim=1)
    return soft_posteriori


def log_softmax(tensor, a):
    """
    Funcion Log_Softmax, utiliza el logaritmo de la softmax.
    :Entrada: Valores en 1D y del tipo tensor.

    """
    return F.log_softmax(tensor, dim=1)
