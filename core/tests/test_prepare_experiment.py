# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
"""Test prepare_experiment function from exp_utils.py"""
import logging
import os
from io import StringIO

import pytest
import yaml
from importlib_resources import files

from core.utils.exp_utils import (
    ckpt_download,
    create_experiment,
    get_experiment_parameters,
)


@pytest.fixture
def config():
    """
    Path of the config file
    """
    return files("core.tests.fixtures").joinpath("config.yaml")


@pytest.fixture
def test_directory(tmpdir):
    """
    Creating a temporary directory where we will put our experiments
    """
    return tmpdir.mkdir("experiment_management")


@pytest.fixture
def user_input_model(monkeypatch):
    """Input chosen model"""
    return monkeypatch.setattr(
        "sys.stdin", StringIO("SSD MobileNet V2 FPNLite 320x320")
    )


def create_test_experiment(experiment_name, test_directory, config, user_input_model):
    """Creates an experiment in temp dir
    args:
        experiment_name: name of the experiment we want to create
        test_directory: temporary folder where we will create experiments,
        config: config file,
        user_input_model: Input chosen model.
    returns:
        new_experiment_name: the new experiment name after adding the current time
    """
    create_experiment(experiment_name, test_directory, config, user_input_model)
    new_experiment_name = os.listdir(test_directory)[0]
    return new_experiment_name


def test_create_experiment_structure(
    test_directory, config, experiment_name, user_input_model
):
    """
    This function will test that the experiment folder structure is created
    """
    create_test_experiment(experiment_name, test_directory, config, user_input_model)
    test_dirs = list(os.walk(test_directory))
    assert set(test_dirs[1][1] + test_dirs[1][2]) == set(
        ["logs", "model_checkpoints", "output_images", "config.yaml"]
    )


def test_create_experiment_logging(
    test_directory, config, experiment_name, user_input_model, caplog
):
    """
    Tests the logging info after successfully creating the experiment.
    """
    with caplog.at_level(logging.INFO):
        new_experiment_name = create_test_experiment(
            experiment_name, test_directory, config, user_input_model
        )
    assert "Experiment {} created!".format(new_experiment_name) in caplog.text
    assert (
        "The path of the output model is {}".format(
            os.path.join(test_directory, new_experiment_name, "model_checkpoints")
        )
        in caplog.text
    )

    assert (
        "The path of the logs directory is {}".format(
            os.path.join(test_directory, new_experiment_name, "logs")
        )
        in caplog.text
    )


def test_create_experiment_invalid_model(
    test_directory, config, experiment_name, monkeypatch
):
    """ Tests the behaviour of the function when the user enter an invalid model name"""
    monkeypatch.setattr("sys.stdin", StringIO("model that does not exist"))
    with pytest.raises(RuntimeError) as pytest_wrapped_e:
        create_experiment(experiment_name, test_directory, config, None)
    assert pytest_wrapped_e.type == RuntimeError


def test_create_experiment_no_config(
    test_directory, experiment_name, caplog, user_input_model
):
    """
    This function will test that the experiment folder will not be created
    if the config file does not exist.
    """
    create_experiment(experiment_name, test_directory, "config.yaml", user_input_model)
    caplog.set_level(logging.ERROR)
    assert "verify the path of your config file!" in caplog.text
    assert not os.path.exists(test_directory.join(experiment_name))


def test_create_experiment_existing_exp(
    test_directory, config, experiment_name, monkeypatch
):
    """
    This function will test that if another experiment has the same name
    the user exit the program if he does not want to retrain the existing model.
    """
    test_directory.mkdir(experiment_name)
    monkeypatch.setattr("sys.stdin", StringIO("n"))

    with pytest.raises(SystemExit) as pytest_wrapped_e:
        create_experiment(experiment_name, test_directory, config, None)
    assert pytest_wrapped_e.type == SystemExit
    assert len(os.listdir(test_directory.join(experiment_name))) == 0


def test_create_experiment_update_config(
    test_directory, config, experiment_name, user_input_model
):
    """
    This function will test that the experiment function will update the config file:
    the output_model_path and the logs_path will be the ones created
    in the new experiment folder.

    """
    new_experiment_name = create_test_experiment(
        experiment_name, test_directory, config, "SSD MobileNet V2 FPNLite 320x320"
    )
    with open(
        os.path.join(test_directory, new_experiment_name, os.path.basename(config)), "r"
    ) as file:
        config_file = yaml.load(file, Loader=yaml.Loader)
    assert config_file["training"]["output_model_path"] == os.path.join(
        test_directory, new_experiment_name, "model_checkpoints"
    )
    assert config_file["training"]["logs_path"] == os.path.join(
        test_directory, new_experiment_name, "logs"
    )
    assert os.path.exists(config_file["training"]["model_config"])
    assert (
        config_file["training"]["checkpoint_path"]
        == "/tmp/fine_tune_checkpoint/"
        + config_file["training"]["model_name"]
        + "/checkpoint/ckpt-0"
    )
    assert config_file["training"]["model_name"].find(" ") == -1
    assert config_file["training"]["model_name"].find("(") == -1
    assert len(config_file["data"]["input_shape"]) == 3


def test_ckpt_download(test_directory):
    """
    tests the download of the model checkpoints un the temporary folder.
    """
    (model_name, _, _, ckpt_finetune) = get_experiment_parameters(
        "SSD MobileNet V2 FPNLite 320x320"
    )
    fine_tune_checkpoint_path = os.path.join(test_directory, model_name)
    os.makedirs(fine_tune_checkpoint_path)
    ckpt_download(ckpt_finetune, fine_tune_checkpoint_path)
    assert os.listdir(fine_tune_checkpoint_path) == ["checkpoint"]
