from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from app.bnn.RNA import train
from typing import Optional, List, Tuple, Dict, Any
import os
import json
import base64
import torch
import pandas as pd
import numpy as np
from app.config import conf_manager
from app.bnn.redneuronal_bay.RedNeuBay import RedNeuBay
from app.bnn.redneuronal_bay.Div_Datos import trat_Dat, trat_Imag
import torchvision
import torchvision.datasets as dset

router = APIRouter()


class BayesianConfig(BaseModel):
    distribution_type: str = Field(
        default="normal", description="Type of prior distribution"
    )
    markov: bool = Field(default=False, description="Use Markov chain for training")
    mu: float = Field(
        default=0.0,
        description="Mean parameter (Normal, ExGaussian, Logistic, LogNormal, Poisson)",
    )
    sigma: float = Field(
        default=1.0,
        description="Standard deviation or scale parameter (Normal, HalfNormal, ExGaussian, LogNormal, GaussianRandomWalk)",
    )
    alpha: float = Field(
        default=1.0,
        description="Alpha parameter (Cauchy, Beta, Gamma, Weibull, BetaBinomial)",
    )
    beta: float = Field(
        default=1.0, description="Beta parameter (Cauchy, Gamma, Weibull, BetaBinomial)"
    )
    lambdaPar: float = Field(default=1.0, description="Lambda parameter (Exponential)")
    nu: float = Field(default=1.0, description="Nu parameter (ChiSquared, ExGaussian)")
    scale: float = Field(default=1.0, description="Scale parameter (Logistic)")
    lower: float = Field(default=0.0, description="Lower bound (Uniform)")
    upper: float = Field(default=1.0, description="Upper bound (Uniform)")
    p: float = Field(
        default=0.5,
        description="Probability parameter (Bernoulli, Binomial, Categorical)",
    )
    n: int = Field(
        default=1,
        description="Number of trials (Binomial, BetaBinomial, DirichletMultinomial, Multinomial)",
    )
    alpha_vector: Optional[str] = Field(
        default=None,
        description="Alpha vector for Dirichlet/DirichletMultinomial (comma-separated string or list)",
    )
    p_vector: Optional[str] = Field(
        default=None,
        description="Probability vector for Multinomial/Categorical (comma-separated string or list)",
    )
    weights: Optional[str] = Field(
        default=None,
        description="Weights for NormalMixture (comma-separated string or list)",
    )
    means: Optional[str] = Field(
        default=None,
        description="Means for NormalMixture (comma-separated string or list)",
    )
    sigmas: Optional[str] = Field(
        default=None,
        description="Sigmas for NormalMixture (comma-separated string or list)",
    )
    k: float = Field(default=1.0, description="AR1 k parameter")
    tau: float = Field(default=1.0, description="AR1 tau parameter")
    rho: float = Field(default=0.5, description="AR1 rho parameter")


class NeuralNetworkParameters(BaseModel):
    alpha: float = Field(default=0.001, description="Learning rate")
    epoch: int = Field(default=20, description="Number of training epochs")
    criteria: str = Field(default="cross_entropy", description="Loss function")
    optimizer: str = Field(default="SGD", description="Optimizer type")
    image_size: Optional[int] = Field(default=None, description="Image size for CNN")
    verbose: bool = Field(default=True, description="Verbose output")
    decay: float = Field(default=0.0, description="Weight decay")
    momentum: float = Field(default=0.9, description="Momentum for SGD")
    image: bool = Field(default=False, description="Whether input is image data")
    FA_ext: Optional[str] = Field(
        default=None, description="External activation function"
    )
    useBayesian: bool = Field(default=False, description="Use Bayesian neural network")
    bayesian_config: Optional[BayesianConfig] = Field(
        default=None, description="Configuration for Bayesian priors"
    )
    save_mod: str = Field(default="ModiR", description="Model save name")
    pred_hot: bool = Field(default=True, description="Use one-hot prediction")
    test_size: float = Field(default=0.2, description="Test set ratio")
    batch_size: int = Field(default=64, description="Batch size")
    cv: bool = Field(default=True, description="Use cross-validation")
    numFolds: int = Field(default=5, description="Number of folds for cross-validation")
    layers: List[str] = Field(
        default=[],
        description="Neural network layers specifications in the form 'Activation_Function_Name(inputs, outputs)'",
    )


class TestModelParameters(BaseModel):
    model_name: Optional[str] = Field(
        default=None, description="Model name to test (defaults to current saved model)"
    )
    use_test_split: bool = Field(
        default=True, description="Use automatic test split from dataset"
    )
    test_size: float = Field(
        default=0.2, description="Test set ratio if using auto split"
    )
    return_predictions: bool = Field(
        default=False, description="Return individual predictions"
    )
    return_confusion_matrix: bool = Field(
        default=True, description="Return confusion matrix"
    )
    generate_plots: bool = Field(
        default=True, description="Generate visualization plots"
    )


class TestResults(BaseModel):
    success: bool
    accuracy: Optional[float] = None
    model_name: Optional[str] = None
    test_samples: Optional[int] = None
    predictions: Optional[List[Dict[str, Any]]] = None
    confusion_matrix: Optional[Dict[str, Any]] = None
    class_metrics: Optional[Dict[str, Dict[str, Any]]] = None
    plots: Optional[Dict[str, str]] = None  # Base64 encoded images
    error_message: Optional[str] = None
    dataset_info: Optional[Dict[str, Any]] = None


# ===============================================================================
# CONFIGURATION HELPER FUNCTIONS
# ===============================================================================


def safe_load_config(file_path: str) -> dict:
    """
    Safely load configuration file that may contain Python boolean values.

    Args:
        file_path (str): Path to the configuration file

    Returns:
        dict: Parsed configuration

    Raises:
        Exception: If file cannot be parsed
    """
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # First try standard JSON parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # If JSON fails, try replacing Python booleans with JSON booleans
            content = content.replace("True", "true").replace("False", "false")
            return json.loads(content)

    except Exception as e:
        raise Exception(f"Could not parse configuration file {file_path}: {str(e)}")


# ===============================================================================
# SUPPORT FUNCTIONS FOR TEST/PREDICT ENDPOINT
# ===============================================================================


def get_available_models() -> List[str]:
    """
    Get list of available trained models from the output directory.

    Returns:
        List[str]: List of available model names
    """
    output_dir = "./app/bnn/output"
    if not os.path.exists(output_dir):
        return []

    models = []
    for file in os.listdir(output_dir):
        # Look for files that start with 'best_' (trained models)
        if (
            file.startswith("best_")
            and not file.endswith(".csv")
            and not file.endswith(".json")
        ):
            models.append(file)

    return models


def validate_model_file(model_name: str) -> Tuple[bool, str, str]:
    """
    Validate that a model file exists and is accessible.

    Args:
        model_name (str): Name of the model to validate

    Returns:
        Tuple[bool, str, str]: (is_valid, full_path, error_message)
    """
    output_dir = "./app/bnn/output"

    # If no model_name provided, try to get from configuration
    if not model_name:
        try:
            parameters_path = "./app/config/nn_parameters.conf"
            if os.path.exists(parameters_path):
                nn_parameters = safe_load_config(parameters_path)
                save_mod = nn_parameters.get("save_mod", "ModiR")
                model_name = f"best_{save_mod}"
            else:
                model_name = "best_ModiR"  # Default fallback
        except Exception as e:
            return False, "", f"Error reading configuration: {str(e)}"

    # Ensure model_name has 'best_' prefix if not provided
    if not model_name.startswith("best_"):
        model_name = f"best_{model_name}"

    full_path = os.path.join(output_dir, model_name)

    if not os.path.exists(full_path):
        available_models = get_available_models()
        return (
            False,
            full_path,
            f"Model '{model_name}' not found. Available models: {available_models}",
        )

    try:
        # Try to load the model to validate it's not corrupted
        torch.load(full_path, map_location="cpu", weights_only=False)
        return True, full_path, ""
    except Exception as e:
        return False, full_path, f"Model file is corrupted or invalid: {str(e)}"


def load_model_safely(model_path: str) -> Tuple[Optional[Dict], str]:
    """
    Safely load a PyTorch model file with error handling.

    Args:
        model_path (str): Path to the model file

    Returns:
        Tuple[Optional[Dict], str]: (model_layers_dict, error_message)
    """
    try:
        # Load the model
        model_layers = torch.load(model_path, map_location="cpu", weights_only=False)

        # Validate that it's a proper model structure (dict of layers)
        if not isinstance(model_layers, dict):
            return None, "Invalid model format: expected dictionary of layers"

        # Check if it has the expected structure
        if len(model_layers) == 0:
            return None, "Model appears to be empty"

        return model_layers, ""

    except Exception as e:
        return None, f"Failed to load model: {str(e)}"


def get_current_dataset_info() -> Tuple[Optional[Dict], str]:
    """
    Get information about the currently configured dataset.

    Returns:
        Tuple[Optional[Dict], str]: (dataset_info, error_message)
    """
    try:
        dataset_name = conf_manager.get_value("data_file")
        has_header = conf_manager.get_value("has_header")

        if not dataset_name:
            return None, "No dataset configured. Please select a dataset first."

        dataset_info = {
            "name": dataset_name,
            "has_header": has_header,
            "is_mnist": dataset_name.lower() == "mnist",
            "file_path": dataset_name if dataset_name != "mnist" else None,
        }

        # Validate dataset exists
        if dataset_name.lower() == "mnist":
            mnist_dir = "./app/data/MNIST"
            if not os.path.exists(mnist_dir):
                return (
                    None,
                    "MNIST dataset not found. Please ensure MNIST data is available.",
                )
        else:
            if not os.path.exists(dataset_name):
                return None, f"Dataset file not found: {dataset_name}"

        return dataset_info, ""

    except Exception as e:
        return None, f"Error getting dataset info: {str(e)}"


def prepare_test_data_from_dataset(
    dataset_info: Dict, test_size: float = 0.2
) -> Tuple[Optional[Tuple], str]:
    """
    Prepare test data from the current dataset configuration.

    Args:
        dataset_info (Dict): Dataset information from get_current_dataset_info
        test_size (float): Proportion of data to use for testing

    Returns:
        Tuple[Optional[Tuple], str]: ((X_test, Y_test, is_image), error_message)
    """
    try:
        if dataset_info["is_mnist"]:
            # Handle MNIST dataset
            transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )

            root = "./app/data"
            try:
                test_set = dset.MNIST(
                    root=root, train=False, transform=transforms, download=False
                )

                # Get test data in the format expected by the predict method
                batch_size = len(test_set)
                test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False,
                    pin_memory=False,
                )

                for inputs, targets in test_loader:
                    X_test, Y_test = inputs, targets
                    break

                return (X_test, Y_test, True), ""

            except Exception as e:
                return None, f"Error loading MNIST dataset: {str(e)}"

        else:
            # Handle regular tabular datasets
            filename = dataset_info["file_path"]
            has_header = dataset_info["has_header"]

            # Read the dataset
            if has_header:
                df_cla = pd.read_csv(filename, header=0)
            else:
                # Determine number of columns and create column names
                sample_df = pd.read_csv(filename, nrows=1, header=None)
                num_columns = len(sample_df.columns)
                names = [f"x{i+1}" for i in range(num_columns - 1)] + ["class"]
                df_cla = pd.read_csv(filename, names=names, header=None)

            # Handle potential ID columns
            first_col = df_cla.columns[0]
            if df_cla[first_col].nunique() == len(df_cla) and (
                first_col.lower() in ["id", "index"] or first_col.startswith("x1")
            ):
                df_cla = df_cla.drop(first_col, axis=1)

            # Ensure the last column is treated as the class/target variable
            class_col = df_cla.columns[-1]
            if class_col.lower() != "class":
                df_cla = df_cla.rename(columns={class_col: "class"})

            # Use the existing data processing function
            _, X_test, Y_test = trat_Dat(
                df_cla=df_cla, test_size=test_size, batch_size=None
            )

            return (X_test, Y_test, False), ""

    except Exception as e:
        return None, f"Error preparing test data: {str(e)}"


def create_neural_network_for_prediction(
    model_layers: Dict, dataset_info: Dict
) -> Tuple[Optional[RedNeuBay], str]:
    """
    Create a RedNeuBay instance configured for prediction with the loaded model.

    Args:
        model_layers (Dict): Loaded model layers
        dataset_info (Dict): Dataset information

    Returns:
        Tuple[Optional[RedNeuBay], str]: (neural_network, error_message)
    """
    try:
        # Get configuration parameters
        parameters_path = "./app/config/nn_parameters.conf"
        default_params = {
            "alpha": 0.001,
            "epoch": 20,
            "criteria": "cross_entropy",
            "optimizer": "sgd",
            "image_size": 784 if dataset_info["is_mnist"] else 1,
            "verbose": False,  # Set to False for prediction
            "decay": 0.0,
            "momentum": 0.9,
            "image": dataset_info["is_mnist"],
            "FA_ext": None,
            "Bay": False,
            "save_mod": "ModiR",
            "pred_hot": False,  # We'll handle prediction manually
            "test_size": 0.2,
            "batch_size": 64,
            "cv": False,
            "Kfold": 5,
        }

        # Try to load saved parameters
        if os.path.exists(parameters_path):
            try:
                saved_params = safe_load_config(parameters_path)
                default_params.update(saved_params)
            except:
                pass  # Use defaults if config is corrupted

        # Create RedNeuBay instance
        rn = RedNeuBay(
            alpha=default_params["alpha"],
            epoch=default_params["epoch"],
            criteria=default_params["criteria"],
            optimizer=default_params["optimizer"],
            image_size=default_params["image_size"],
            verbose=default_params["verbose"],
            decay=default_params["decay"],
            momentum=default_params["momentum"],
            image=default_params["image"],
            FA_ext=default_params["FA_ext"],
            Bay=default_params["Bay"],
            save_mod=default_params["save_mod"],
            pred_hot=default_params["pred_hot"],
            test_size=default_params["test_size"],
            batch_size=default_params["batch_size"],
            cv=default_params["cv"],
            Kfold=default_params["Kfold"],
        )

        return rn, ""

    except Exception as e:
        return None, f"Error creating neural network for prediction: {str(e)}"


def safe_predict_with_model(
    rn: RedNeuBay,
    model_layers: Dict,
    X_test,
    Y_test,
    is_image: bool,
    return_predictions: bool = False,
) -> Tuple[Optional[Dict], str]:
    """
    Safely execute prediction using the RedNeuBay predict method.

    Args:
        rn (RedNeuBay): Neural network instance
        model_layers (Dict): Loaded model layers
        X_test: Test input data
        Y_test: Test target data
        is_image (bool): Whether the data is image data
        return_predictions (bool): Whether to return individual predictions

    Returns:
        Tuple[Optional[Dict], str]: (prediction_results, error_message)
    """
    try:
        # Determine image size
        image_size = 784 if is_image else 1

        # Execute prediction using the existing predict method
        # The predict method handles the actual prediction logic
        result = rn.predict(
            mod=model_layers,
            x=X_test,
            y=Y_test,
            img=is_image,
            image_size=image_size,
            target=True,
        )

        # Calculate accuracy manually since predict method prints but doesn't return it
        with torch.no_grad():
            if is_image:
                enput = X_test.view(-1, image_size)
            else:
                enput = torch.FloatTensor(X_test)
                Y_test = torch.Tensor(Y_test)

            # Forward pass through all layers
            for i in range(len(model_layers)):
                layer = model_layers[i]
                a = 1
                output = layer.funcion_activacion(
                    torch.add(torch.matmul(enput, layer.weights), layer.bias), a
                )
                output = torch.FloatTensor(output)
                enput = output

            # Get predictions
            _, pred = torch.max(output, 1)

            # Calculate accuracy
            n_total_row = len(Y_test)
            accuracy = torch.sum(pred == Y_test).float() / n_total_row
            accuracy = float(accuracy.numpy())

            prediction_results = {
                "accuracy": round(accuracy * 100.0, 2),
                "total_samples": n_total_row,
                "correct_predictions": int(torch.sum(pred == Y_test).item()),
                "predictions": pred.numpy().tolist(),  # Always include for confusion matrix
                "true_labels": Y_test.numpy().tolist(),  # Always include for confusion matrix
                "return_predictions": return_predictions,  # Flag to control response inclusion
            }

            return prediction_results, ""

    except Exception as e:
        return None, f"Error during prediction: {str(e)}"


# ===============================================================================
# END SUPPORT FUNCTIONS
# ===============================================================================


def save_parameters_to_conf(params: dict):
    """
    Save neural network parameters to the configuration file.

    Args:
        params (dict): Dictionary containing the neural network parameters
    """
    config_path = "./app/config/nn_parameters.conf"
    with open(config_path, "w") as f:
        f.write("{\n")
        # Write each parameter in Python literal format
        for i, (key, value) in enumerate(params.items()):
            if value is None:
                formatted_value = "None"
            elif isinstance(value, bool):
                formatted_value = str(value)
            elif isinstance(value, str):
                formatted_value = f'"{value}"'
            elif isinstance(value, list):
                if not value:  # Empty list
                    formatted_value = "[]"
                elif key == "layers":
                    # Format layers list with each element in quotes
                    layer_items = [f'"{layer}"' for layer in value]
                    formatted_value = f'[{", ".join(layer_items)}]'
                else:
                    formatted_value = str(value).replace("'", "")
            elif isinstance(value, dict):
                # Handle nested dictionaries (like bayesian_config)
                formatted_items = []
                for k, v in value.items():
                    if isinstance(v, str):
                        formatted_items.append(f'"{k}": "{v}"')
                    else:
                        formatted_items.append(f'"{k}": {v}')
                formatted_value = "{" + ", ".join(formatted_items) + "}"
            else:
                formatted_value = str(value)

            # Add comma for all items except the last one
            comma = "," if i < len(params) - 1 else ""
            f.write(f'    "{key}": {formatted_value}{comma}\n')

        f.write("}\n")


@router.get("/results")
async def get_results():
    """
    Obtiene los resultados del entrenamiento de la red neuronal, incluyendo
    el JSON almacenado en results.json y los archivos de imagen (PNG) codificados en Base64.

    Returns:
        dict: Objeto con los datos JSON y las imágenes codificadas en base64.
    """
    try:
        # Load JSON results from file
        results_path = "./app/bnn/output/results.json"
        if not os.path.exists(results_path):
            raise HTTPException(status_code=404, detail="Results file not found.")
        with open(results_path, "r") as f:
            results_data = json.load(f)

        # Load regular PNG images from the plots folder
        images_folder = "./app/data/plots"
        images = {}
        if os.path.exists(images_folder):
            for file in os.listdir(images_folder):
                if file.endswith(".png"):
                    file_path = os.path.join(images_folder, file)
                    with open(file_path, "rb") as img_file:
                        encoded_image = base64.b64encode(img_file.read()).decode(
                            "utf-8"
                        )
                        images[file] = encoded_image

        # Load CV-specific images from the cv subfolder
        cv_images_folder = os.path.join(images_folder, "cv")
        if os.path.exists(cv_images_folder):
            for file in os.listdir(cv_images_folder):
                if file.endswith(".png"):
                    file_path = os.path.join(cv_images_folder, file)
                    with open(file_path, "rb") as img_file:
                        encoded_image = base64.b64encode(img_file.read()).decode(
                            "utf-8"
                        )
                        # Add to images with cv/ prefix to distinguish them
                        images[f"cv/{file}"] = encoded_image

        # Format all numerical results for better readability
        formatted_results = {}

        # Handle accuracy and epoch specially
        if "accuracy" in results_data:
            formatted_results["Accuracy"] = f"{results_data['accuracy']}%"
        if "epoch" in results_data:
            formatted_results["Best Epoch"] = str(results_data["epoch"])

        # Format class frequencies
        if "overall_class_frequency" in results_data:
            class_freq = []
            for class_name, count in results_data["overall_class_frequency"].items():
                class_freq.append(f"{class_name}: {count}")
            formatted_results["Class Frequency"] = ", ".join(class_freq)

        if "image_class_frequency" in results_data:
            img_class_freq = []
            for class_name, count in results_data["image_class_frequency"].items():
                img_class_freq.append(f"{class_name}: {count}")
            formatted_results["Image Class Frequency"] = ", ".join(img_class_freq)

        # Handle CV fold class frequencies
        if "class_frequency" in results_data:
            for fold, freqs in results_data["class_frequency"].items():
                fold_freqs = []
                for class_name, count in freqs.items():
                    fold_freqs.append(f"{class_name}: {count}")
                formatted_results[f"Class Frequency ({fold})"] = ", ".join(fold_freqs)

        # Return both the formatted results and encoded images
        return {
            "text_results": formatted_results,
            "raw_results": results_data,
            "images": images,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/normal")
async def train_rna(params: NeuralNetworkParameters):
    """
    Entrena una red neuronal normal con los parámetros proporcionados.

    Los parámetros se envían desde el frontend y se pasan al método de entrenamiento.
    También se guardan en el archivo de configuración para uso futuro.
    """
    try:
        # Convert Pydantic model to dictionary
        train_params = params.model_dump()

        # Extract bayesian config if available
        bayesian_config = train_params.pop("bayesian_config", None)

        # Rename parameters for the train function
        train_params["Bay"] = train_params.pop("useBayesian")
        train_params["Kfold"] = train_params.pop("numFolds")

        # Save the full config (including bayesian_config) to the conf file
        save_parameters = train_params.copy()
        if bayesian_config:
            save_parameters["bayesian_config"] = bayesian_config
        save_parameters_to_conf(save_parameters)

        # Add individual bayesian parameters to train_params if bayesian_config exists
        if bayesian_config:
            train_params["lambdaPar"] = bayesian_config.get("lambdaPar", 0.1)
        train(**train_params)
        return {
            "message": "Entrenamiento de red neuronal completado con éxito y parámetros guardados."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download")
async def download_best_model():
    """
    Endpoint to download the best model of the training
    Returns the best saved model as a downloadable file
    """
    try:
        # Get the model name from the results.json or use default
        parameters_path = "./app/config/nn_parameters.conf"
        model_name = "ModiR"  # Default model name

        if os.path.exists(parameters_path):
            try:
                nn_parameters = safe_load_config(parameters_path)
                model_name = nn_parameters.get("save_mod", "ModiR")
            except:
                pass

        # Construct the best model filename
        best_model_filename = f"best_{model_name}"
        model_path = f"./app/bnn/output/{best_model_filename}"

        # Check if the model file exists
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=404, detail=f"Model file not found: {best_model_filename}"
            )

        # Return the file as a download
        return FileResponse(
            path=model_path,
            filename=f"{best_model_filename}.pth",
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={best_model_filename}.pth"
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error downloading model: {str(e)}"
        )


@router.post("/test", response_model=TestResults)
async def test_model(params: TestModelParameters) -> TestResults:
    """
    Test a trained neural network model using the predict method.

    This endpoint loads a saved model and tests it against test data,
    returning accuracy metrics, confusion matrix, and optional predictions.

    Args:
        params (TestModelParameters): Testing configuration parameters

    Returns:
        TestResults: Comprehensive test results including accuracy and metrics

    Raises:
        HTTPException: If model loading, data preparation, or prediction fails
    """
    try:
        # Step 1: Validate and load the model
        is_valid, model_path, error_msg = validate_model_file(params.model_name)
        if not is_valid:
            return TestResults(
                success=False, error_message=f"Model validation failed: {error_msg}"
            )

        model_layers, load_error = load_model_safely(model_path)
        if model_layers is None:
            return TestResults(
                success=False, error_message=f"Model loading failed: {load_error}"
            )

        # Step 2: Get dataset information
        dataset_info, dataset_error = get_current_dataset_info()
        if dataset_info is None:
            return TestResults(
                success=False,
                error_message=f"Dataset configuration error: {dataset_error}",
            )

        # Step 3: Prepare test data
        test_data_result, prep_error = prepare_test_data_from_dataset(
            dataset_info, params.test_size
        )
        if test_data_result is None:
            return TestResults(
                success=False,
                error_message=f"Test data preparation failed: {prep_error}",
            )

        X_test, Y_test, is_image = test_data_result

        # Step 4: Create neural network instance for prediction
        rn, nn_error = create_neural_network_for_prediction(model_layers, dataset_info)
        if rn is None:
            return TestResults(
                success=False,
                error_message=f"Neural network creation failed: {nn_error}",
            )

        # Step 5: Execute prediction
        prediction_results, pred_error = safe_predict_with_model(
            rn, model_layers, X_test, Y_test, is_image, params.return_predictions
        )
        if prediction_results is None:
            return TestResults(
                success=False,
                error_message=f"Prediction execution failed: {pred_error}",
            )

        # Step 6: Prepare response
        model_name = os.path.basename(model_path)

        response = TestResults(
            success=True,
            accuracy=prediction_results["accuracy"],
            model_name=model_name,
            test_samples=prediction_results["total_samples"],
            dataset_info=dataset_info,
        )

        # Add predictions if requested
        if params.return_predictions:
            predictions_list = []
            for i, (pred, true) in enumerate(
                zip(
                    prediction_results["predictions"], prediction_results["true_labels"]
                )
            ):
                predictions_list.append(
                    {
                        "index": i,
                        "predicted": int(pred),
                        "actual": int(true),
                        "correct": int(pred) == int(true),
                    }
                )
            response.predictions = predictions_list

        # Add confusion matrix if requested
        if params.return_confusion_matrix:
            try:
                from sklearn.metrics import confusion_matrix, classification_report

                # Check if we have predictions and true labels
                if (
                    prediction_results.get("predictions") is not None
                    and prediction_results.get("true_labels") is not None
                ):

                    pred_array = np.array(prediction_results["predictions"])
                    true_array = np.array(prediction_results["true_labels"])

                    cm = confusion_matrix(true_array, pred_array)

                    # Create confusion matrix dictionary
                    unique_labels = sorted(list(set(true_array)))
                    cm_dict = {
                        "matrix": cm.tolist(),
                        "labels": unique_labels,
                        "size": len(unique_labels),
                    }
                    response.confusion_matrix = cm_dict

                    # Add class-wise metrics
                    try:
                        class_report = classification_report(
                            true_array, pred_array, output_dict=True
                        )
                        response.class_metrics = class_report
                    except Exception as e:
                        print(f"Warning: Could not generate classification report: {e}")
                else:
                    print(
                        "Warning: No predictions available for confusion matrix generation"
                    )

            except Exception as e:
                print(f"Warning: Could not generate confusion matrix: {e}")

        # Add plots if requested
        if params.generate_plots:
            try:
                # Check for existing plot files
                plots_folder = "./app/data/plots"
                plot_files = {}

                if os.path.exists(plots_folder):
                    for file in os.listdir(plots_folder):
                        if file.endswith(".png"):
                            file_path = os.path.join(plots_folder, file)
                            try:
                                with open(file_path, "rb") as img_file:
                                    encoded_image = base64.b64encode(
                                        img_file.read()
                                    ).decode("utf-8")
                                    plot_files[file] = encoded_image
                            except Exception as e:
                                print(f"Warning: Could not encode plot {file}: {e}")

                if plot_files:
                    response.plots = plot_files

            except Exception as e:
                print(f"Warning: Could not load plot files: {e}")

        return response

    except Exception as e:
        # Catch any unexpected errors
        return TestResults(
            success=False,
            error_message=f"Unexpected error during model testing: {str(e)}",
        )


@router.get("/models")
async def list_available_models():
    """
    Get a list of all available trained models.

    Returns:
        dict: Dictionary containing available models and current configuration
    """
    try:
        available_models = get_available_models()

        # Get current model from configuration
        current_model = None
        try:
            parameters_path = "./app/config/nn_parameters.conf"
            if os.path.exists(parameters_path):
                nn_parameters = safe_load_config(parameters_path)
                save_mod = nn_parameters.get("save_mod", "ModiR")
                current_model = f"best_{save_mod}"
        except:
            pass

        # Get dataset info
        dataset_info, _ = get_current_dataset_info()

        return {
            "available_models": available_models,
            "current_model": current_model,
            "current_dataset": dataset_info["name"] if dataset_info else None,
            "total_models": len(available_models),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")
