""" Create new folder for every new experiment """
import datetime
import logging
import os
import re
import shutil
import sys
from typing import Any, List, Tuple

import wget
import yaml
from cachepath import CachePath
from importlib_resources import files

import core.models.model_configs

models_dict = {
    "SSD EfficientDet D0 512x512": "ssd_efficientdet_d0_512x512_coco17_tpu-32",
    "SSD EfficientDet D1 640x640": "ssd_efficientdet_d1_640x640_coco17_tpu-32",
    "SSD EfficientDet D2 768x768": "ssd_efficientdet_d2_768x768_coco17_tpu-32",
    "SSD EfficientDet D3 896x896": "ssd_efficientdet_d3_896x896_coco17_tpu-32",
    "SSD EfficientDet D4 1024x1024": "ssd_efficientdet_d4_1024x1024_coco17_tpu-32",
    "SSD EfficientDet D5 1280x1280": "ssd_efficientdet_d5_1280x1280_coco17_tpu-32",
    "SSD EfficientDet D6 1280x1280": "ssd_efficientdet_d6_1408x1408_coco17_tpu-32",
    "SSD EfficientDet D7 1536x1536": "ssd_efficientdet_d7_1536x1536_coco17_tpu-32",
    "SSD MobileNet V1 FPN 640x640": "ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8",
    "SSD MobileNet V2 FPNLite 320x320": "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8",
    "SSD MobileNet V2 FPNLite 640x640": "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8",
    "SSD ResNet50 V1 FPN 640x640 (RetinaNet50)": (
        "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8"
    ),
    "SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)": (
        "ssd_resnet50_v1_fpn_1024x1024" "_coco17_tpu-8"
    ),
    "SSD ResNet101 V1 FPN 640x640 (RetinaNet101)": (
        "ssd_resnet101_v1_fpn_640x640_coco17_tpu-8"
    ),
    "SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)": (
        "ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8"
    ),
    "SSD ResNet152 V1 FPN 640x640 (RetinaNet152)": (
        "ssd_resnet152_v1_fpn_640x640_coco17_tpu-8"
    ),
    "SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)": (
        "ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8"
    ),
}


def ckpt_download(ckpt: str, fine_tune_checkpoint_path: str):
    """
    Download the checkpoints of the pretrained model in temporary folder.
    args:
        ckpt: the checkpoints we want to download.
        fine_tune_checkpoint_path: the temporary folder where
                            the checkpoints will be downloaded.
    """
    ckpt_url = (
        "http://download.tensorflow.org/models/object_detection/tf2/20200711/"
        + ckpt
        + ".tar.gz"
    )
    wget.download(ckpt_url, str(fine_tune_checkpoint_path))
    shutil.unpack_archive(
        os.path.join(fine_tune_checkpoint_path, ckpt + ".tar.gz"),
        fine_tune_checkpoint_path,
    )
    shutil.move(
        os.path.join(fine_tune_checkpoint_path, ckpt, "checkpoint"),
        fine_tune_checkpoint_path,
    )
    os.remove(os.path.join(fine_tune_checkpoint_path, ckpt + ".tar.gz"))
    shutil.rmtree(os.path.join(fine_tune_checkpoint_path, ckpt))


def get_experiment_parameters(model: str) -> Tuple[str, List, str, str]:
    """
    Extract experiment parameters from the chosen model.
    args:
        model: the model we want to fine tune.
    returns:
        model_name: cleaned name of the model (without spaces and parenthesis),
        input_shape: the input shape of the model,
        model_config: the configuration of the chosen model,
        ckpt_finetune: the chekpoints name of the model
    """
    model_name = re.sub(r"\([\w\d]+\)", "", model)
    model_name = model_name.lower().strip().replace(" ", "_")
    input_shape = re.findall(r"_(\d{,4})x(\d{,4})", model_name)[0]
    input_shape = [int(dim) for dim in input_shape] + [3]
    model_config = str(
        files(core.models.model_configs).joinpath(models_dict[model] + ".config")
    )
    if "efficientdet" in models_dict[model]:
        ckpt_finetune = "_".join(
            [models_dict[model].split("_")[i] for i in [1, 2, 4, 5]]
        )
    else:
        ckpt_finetune = models_dict[model]
    return model_name, input_shape, model_config, ckpt_finetune


# pylint: disable=too-many-statements
def create_experiment(
    experiment_name: str, experiments_dir: str, config_file: str, model: str
) -> Any:
    """
    This function helps to keep track of models experiments.
    It creates a new experiment folder which contains the corresponding config file,
    a directory for the output model and a directory for the tensorboard logs.

    args:
        experiment_name: name of the new experiment,
        experiments_dir: the directory that contains all the created experiments,
        config_file: configuration file that contains all the information related
                to the led experience.
        model: the pre-trained model full name.
    """
    # pylint: disable=too-many-locals
    # the number is reasonable in our case.
    root = experiments_dir
    experiment_path = os.path.join(root, experiment_name)
    file_exists = os.path.isfile(config_file)
    experiment_exists = os.path.exists(experiment_path)

    if experiment_exists:
        retrain = input(
            "You've already created this experiment!"
            " Do you want to retrain/train the existing model? (y/n) "
        )
        if retrain in ("y", "yes"):
            return os.path.join(experiment_path, os.path.basename(config_file))
        sys.exit()
    elif not file_exists:
        logging.error("verify the path of your config file!")
        return None
    elif (not experiment_exists) and file_exists:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = experiment_name + "_" + current_time
        experiment_path = os.path.join(root, experiment_name)
        if model == None:

            print("*" * 70)
            model = input(
                "Please choose one of those pre-trained models to use: \n"
                + "*" * 70
                + "\n"
                + "\n".join(list(models_dict.keys()))
                + "\n"
                + "*" * 70
                + "\n"
            )

        if model not in models_dict.keys():
            raise RuntimeError("The chosen model is not in the proposed list")
        (
            model_name,
            input_shape,
            model_config,
            ckpt_finetune,
        ) = get_experiment_parameters(model)

        logs_path = os.path.join(experiment_path, "logs")
        model_path = os.path.join(experiment_path, "model_checkpoints")
        fine_tune_checkpoint_path = CachePath("fine_tune_checkpoint", model_name)
        config_path = os.path.join(experiment_path, "config.yaml")
        output_images_path = os.path.join(
            experiments_dir, experiment_name, "output_images"
        )
        os.makedirs(experiment_path)
        logging.info("Experiment %s created!", experiment_name)
        os.makedirs(output_images_path)

        os.makedirs(logs_path)
        os.makedirs(model_path)
        shutil.copy(config_file, experiment_path)
        # download checkpoints if they don't exist
        if not os.path.exists(os.path.join(fine_tune_checkpoint_path, "checkpoint")):
            logging.info("Download the checkpoints...")
            if os.path.exists(fine_tune_checkpoint_path):
                shutil.rmtree(fine_tune_checkpoint_path)
            os.makedirs(fine_tune_checkpoint_path)
            ckpt_download(ckpt_finetune, fine_tune_checkpoint_path)
            logging.info("\n Checkpoints downloaded!")
        else:
            pass
        # update the config file for the training
        with open(config_path, "r") as file:
            parameters = yaml.safe_load(file)
        parameters["training"]["model_name"] = model_name
        parameters["training"]["output_model_path"] = model_path
        parameters["training"]["logs_path"] = logs_path
        parameters["training"]["model_config"] = model_config
        parameters["training"]["checkpoint_path"] = os.path.join(
            fine_tune_checkpoint_path, "checkpoint", "ckpt-0"
        )
        parameters["data"]["input_shape"] = input_shape
        with open(config_path, "w") as conf:
            yaml.dump(parameters, conf)
        logging.info("The path of the output model is %s", model_path)
        logging.info("The path of the logs directory is %s", logs_path)
        return config_path
    else:
        return None
