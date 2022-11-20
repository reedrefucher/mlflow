"""
The ``mlflow.tflite`` module provides an API for logging and loading TensorFlow lite models.
This module exports TensorFlow models with the following flavors:

TensorFlow Lite (native) format
    This is the main flavor that can be loaded back into TensorFlow Lite.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

import logging
import os

import numpy
import tensorflow
import pandas
import yaml
from packaging.version import Version

import mlflow
from mlflow import pyfunc, MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.annotations import keyword_only
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "tflite"
_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """

    pip_deps = [_get_pinned_requirement("tensorflow")]

    # tensorflow >= 2.6.0 requires keras:
    # https://github.com/tensorflow/tensorflow/blob/v2.6.0/tensorflow/tools/pip_package/setup.py#L106
    # To prevent a different version of keras from being installed by tensorflow when creating
    # a serving environment, add a pinned requirement for keras
    if Version(tensorflow.__version__) >= Version("2.6.0"):
        pip_deps.append(_get_pinned_requirement("keras"))

    return pip_deps


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@keyword_only
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    tflite_model,
    artifact_path,
    conda_env=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Log a Tensorflow lite model as an MLflow artifact for the current run.
    :param tflite_model: tflite_model (a byte form instance of
    `tensorflow.lite.python.interpreter.Interpreter`_) to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this describes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.8.8',
                                'pip': [
                                    'tensorflow==2.4.1'
                                ]
                            ]
                        }
    :param registered_model_name: (Experimental) If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.

    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.tflite,
        registered_model_name=registered_model_name,
        tflite_model=tflite_model,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
    )


@keyword_only
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    self,
    tflite_model,
    path,
    conda_env=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Save an tflite model to a path on the local file system.

    :param tflite_model: tflite_model (a byte form instance of
    `tensorflow.lite.python.interpreter.Interpreter`_) to be saved.
    :param path: Local path where the model is to be saved. :param conda_env: Either a dictionary
    representation of a Conda environment or the path to a Conda environment yaml file.
    If provided, this describes the environment this model should be run in.
    At minimum, it should specify the dependencies contained in
    :func:`get_default_conda_env()`. If ``None``, the default :func:`get_default_conda_env()`
    environment is added to the model.
    The following is an *example* dictionary representation of a Conda environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.8.8',
                                'pip': [
                                    'tensorflow==2.4.1'
                                ]
                            ]
                        }

    :param mlflow_model: MLflow model configuration to which to add the ``tflite`` flavor.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset).

    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}

    """

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)
    # TODO: add block to validate model

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    model_data_subpath = "model.tflite"
    model_data_path = os.path.join(path, model_data_subpath)

    # Save a tflite model
    with open(model_data_path, "wb") as f:
        f.write(tflite_model)

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path,
                FLAVOR_NAME,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    mlflow_model.add_flavor(
        self.FLAVOR_NAME, tf_version=tensorflow.__version__, data=model_data_subpath
    )
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.tflite",
        data=model_data_subpath,
        env=_CONDA_ENV_FILE_NAME,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def load_model(self, model_uri):
    """
    Load a Tensorflow lite model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.

    :return: An TFLite model (an instance of `tensorflow.lite.python.interpreter.Interpreter`_)
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(
        model_path=local_model_path, flavor_name=self.FLAVOR_NAME
    )
    tflite_model_file_path = os.path.join(local_model_path, flavor_conf.get("data", "model.tflite"))
    return _load_model(model_path=tflite_model_file_path)


def _load_model(model_path):
    """
    Load a specified TensorFlow model from model path and return instance of
    tensorflow.lite.Interpreter
    """
    interpreter = tensorflow.lite.Interpreter(model_path=model_path)
    return interpreter


def _load_pyfunc(model_path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``. This function loads an MLflow
    model with the TensorFlow Lite flavor into a new tensorflow.lite.Interpreter instance
    and exposes it behind the ``pyfunc.predict`` interface.
    """
    return _TFLiteInterpreterWrapper(model_path=model_path)


class _TFLiteInterpreterWrapper(tensorflow.lite.Interpreter):
    """
    Wrapper class that exposes a TensorFlow Lite model for inference via a ``predict`` function
    such that:
        ``predict(data: numpy.array) -> numpy.array``.
        ``predict(data: dict) -> numpy.array``.
        ``predict(data: pandas.core.frame.DataFrame) -> numpy.array``.
    """

    def __init__(self, model_path=None, model_content=None, **kwargs):
        super().__init__(model_path, model_content, **kwargs)
        self.allocate_tensors()
        self.inputs = [
            (inp["name"], inp["dtype"], inp["shape"], inp["index"])
            for inp in self.get_input_details()
        ]
        self.outputs = [
            (outp["name"], outp["dtype"], outp["shape"], outp["index"])
            for outp in self.get_output_details()
        ]

    def _cast_input_types(self, feeds):
        """
        This helper ensures the data to be cast into expected format for
        """
        for input_name, input_type, _, _ in self.inputs:
            feed = feeds.get(input_name)
            if feed is not None and feed.dtype != input_type:
                feeds[input_name] = feed.astype(input_type)
        return feeds

    def predict(self, data):
        """

        :param data: Either a pandas DataFrame, numpy.ndarray or a dictionary.

            Dictionary input is expected to be a valid TFLite model feed dictionary.
            Numpy array input is supported
            Pandas DataFrame is converted to TFlite inputs as follows:
                - If the underlying TFLite model only defines a *single* input tensor, the
                DataFrame's values are converted to a NumPy array representation using the
                `DataFrame.values()
                <https://pandas.pydata.org/pandas-docs/stable/reference/api/
                 pandas.DataFrame.values.html#pandas.DataFrame.values>`_ method.
                 - If the underlying TFlite model defines *multiple* input tensors, each column
                 of the DataFrame is converted to a NumPy array representation.


        :return: Model predictions in NumPy array representation

        """

        if isinstance(data, dict):
            feed_dict = data
        elif isinstance(data, numpy.ndarray):
            # NB: We do allow scoring with a single tensor (ndarray) in order to be compatible with
            # supported pyfunc inputs iff the model has a single input. The passed tensor is
            # assumed to be the first input.
            if len(self.inputs) != 1:
                inputs = [x[0] for x in self.inputs]
                raise MlflowException(
                    "Unable to map numpy array input to the expected model"
                    "input. "
                    "Numpy arrays can only be used as input for MLflow Tflite"
                    "models that have a single input. This model requires "
                    "{0} inputs. Please pass in data as either a "
                    "dictionary or a DataFrame with the following tensors"
                    ": {1}.".format(len(self.inputs), inputs)
                )
            feed_dict = {self.inputs[0][0]: data}

        elif isinstance(data, pandas.DataFrame):
            if len(self.inputs) > 1:
                feed_dict = {name: data[[name]].values for (name, _, _, _) in self.inputs}
            else:
                feed_dict = {self.inputs[0][0]: data.values}

        else:
            raise TypeError(
                "Input should be a dictionary or a numpy array or a pandas.DataFrame,"
                "got '{}'".format(type(data))
            )

        feed_dict = self._cast_input_types(feed_dict)

        # return(feed_dict)

        if len(self.inputs) > 1:
            assert len({inp.shape[0] for inp in feed_dict.values()}) == 1, (
                "provided inputs have different batch"
                "size, please, ensure that batch size"
                " is equal across different inputs "
            )
        output_data = []
        for i in range(len(feed_dict[self.inputs[0][0]])):
            for inp in self.inputs:
                self.set_tensor(inp[3], feed_dict[inp[0]][i : (i + 1)])
            self.invoke()
            output_data.extend(self.get_tensor(self.outputs[0][3]))

        return numpy.array(output_data)
