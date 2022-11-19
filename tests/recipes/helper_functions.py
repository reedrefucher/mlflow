import mlflow
import os
import pathlib
import random
import shutil
import string
import sys
from typing import Generator

from contextlib import contextmanager
from mlflow.recipes.utils.execution import _MLFLOW_RECIPES_EXECUTION_DIRECTORY_ENV_VAR
from mlflow.recipes.steps.split import _OUTPUT_TEST_FILE_NAME, _OUTPUT_VALIDATION_FILE_NAME
from mlflow.recipes.step import BaseStep
from mlflow.utils.file_utils import TempDir
from pathlib import Path
from sklearn.datasets import load_diabetes, load_iris
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

import pytest

RECIPE_EXAMPLE_PATH_ENV_VAR_FOR_TESTS = "_RECIPE_EXAMPLE_PATH"
RECIPE_EXAMPLE_PATH_FROM_MLFLOW_ROOT = "examples/recipes/regression"

## Methods
def get_random_id(length=6):
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))


def setup_model_and_evaluate(tmp_recipe_exec_path: Path):
    split_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "split", "outputs")
    split_step_output_dir.mkdir(parents=True)
    X, y = load_diabetes(as_frame=True, return_X_y=True)
    validation_df = X.assign(y=y).sample(n=50, random_state=9)
    validation_df.to_parquet(split_step_output_dir.joinpath(_OUTPUT_VALIDATION_FILE_NAME))
    test_df = X.assign(y=y).sample(n=100, random_state=42)
    test_df.to_parquet(split_step_output_dir.joinpath(_OUTPUT_TEST_FILE_NAME))

    run_id, model = train_and_log_model()
    train_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "train", "outputs")
    train_step_output_dir.mkdir(parents=True)
    train_step_output_dir.joinpath("run_id").write_text(run_id)
    output_model_path = train_step_output_dir.joinpath("model")
    if os.path.exists(output_model_path) and os.path.isdir(output_model_path):
        shutil.rmtree(output_model_path)
    mlflow.sklearn.save_model(model, output_model_path)

    evaluate_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "evaluate", "outputs")
    evaluate_step_output_dir.mkdir(parents=True)

    register_step_output_dir = tmp_recipe_exec_path.joinpath("steps", "register", "outputs")
    register_step_output_dir.mkdir(parents=True)
    return evaluate_step_output_dir, register_step_output_dir


def train_and_log_model(is_dummy=False):
    mlflow.set_experiment("demo")
    with mlflow.start_run() as run:
        X, y = load_diabetes(as_frame=True, return_X_y=True)
        if is_dummy:
            model = DummyRegressor(strategy="constant", constant=42)
        else:
            model = LinearRegression()
        fitted_model = model.fit(X, y)
        mlflow.sklearn.log_model(fitted_model, artifact_path="train/model")
        return run.info.run_id, fitted_model


def train_and_log_classification_model(is_dummy=False):
    mlflow.set_experiment("demo")
    with mlflow.start_run() as run:
        X, y = load_iris(as_frame=True, return_X_y=True)
        if is_dummy:
            model = DummyClassifier(strategy="constant", constant=42)
        else:
            model = LogisticRegression()
        fitted_model = model.fit(X, y)
        mlflow.sklearn.log_model(fitted_model, artifact_path="train/model")
        return run.info.run_id, fitted_model


def train_log_and_register_model(model_name, is_dummy=False):
    run_id, _ = train_and_log_model(is_dummy)
    runs_uri = "runs:/{}/train/model".format(run_id)
    mv = mlflow.register_model(runs_uri, model_name)
    return f"models:/{mv.name}/{mv.version}"


## Fixtures
@pytest.fixture
def enter_recipe_example_directory():
    recipe_example_path = os.environ.get(RECIPE_EXAMPLE_PATH_ENV_VAR_FOR_TESTS)
    if recipe_example_path is None:
        mlflow_repo_root_directory = pathlib.Path(mlflow.__file__).parent.parent
        recipe_example_path = mlflow_repo_root_directory / RECIPE_EXAMPLE_PATH_FROM_MLFLOW_ROOT

    with chdir(recipe_example_path):
        yield recipe_example_path


@pytest.fixture
def enter_test_recipe_directory(enter_recipe_example_directory):
    recipe_example_root_path = enter_recipe_example_directory

    with TempDir(chdr=True) as tmp:
        test_recipe_path = tmp.path("test_recipe")
        shutil.copytree(recipe_example_root_path, test_recipe_path)
        os.chdir(test_recipe_path)
        yield os.getcwd()


@pytest.fixture
def tmp_recipe_exec_path(monkeypatch, tmp_path) -> Path:
    path = tmp_path.joinpath("recipe_execution")
    path.mkdir(parents=True)
    monkeypatch.setenv(_MLFLOW_RECIPES_EXECUTION_DIRECTORY_ENV_VAR, str(path))
    yield path
    shutil.rmtree(path)


@pytest.fixture
def tmp_recipe_root_path(tmp_path) -> Path:
    path = tmp_path.joinpath("recipe_root")
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path)


@pytest.fixture
def clear_custom_metrics_module_cache():
    key = "steps.custom_metrics"
    if key in sys.modules:
        del sys.modules[key]


@pytest.fixture
def registry_uri_path(tmp_path) -> Path:
    path = tmp_path.joinpath("registry.db")
    db_url = "sqlite:///%s" % path
    yield db_url
    mlflow.set_registry_uri("")


@contextmanager
def chdir(directory_path):
    og_dir = os.getcwd()
    try:
        os.chdir(directory_path)
        yield
    finally:
        os.chdir(og_dir)


class BaseStepImplemented(BaseStep):
    def _run(self, output_directory):
        pass

    def _inspect(self, output_directory):
        pass

    def clean(self):
        pass

    @classmethod
    def from_recipe_config(cls, recipe_config, recipe_root):
        pass

    @property
    def name(self):
        pass

    def _validate_and_apply_step_config(self):
        pass

    def step_class(self):
        pass


def list_all_artifacts(
    tracking_uri: str, run_id: str, path: str = None
) -> Generator[str, None, None]:
    artifacts = mlflow.tracking.MlflowClient(tracking_uri).list_artifacts(run_id, path)
    for artifact in artifacts:
        if artifact.is_dir:
            yield from list_all_artifacts(tracking_uri, run_id, artifact.path)
        else:
            yield artifact.path
