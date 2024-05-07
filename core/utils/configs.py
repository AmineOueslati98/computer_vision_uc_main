# # Copyright 2021 INSTADEEP.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ======================================================================
""" load the YAML file and transforms it into a class instance
for easier management of parameters"""
# pylint: disable=too-many-instance-attributes

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yaml

from core.ocr.config import OcrConfig


@dataclass
class ExperimentConfig:
    """Class that defines an experiment config."""

    data: "DataConfig"
    augmentation: "AugmentationConfig"
    training: "TrainingConfig"
    inference: "InferenceConfig"
    ocr: "OcrConfig"

    def is_valid(self) -> bool:
        """ verify if the configuration fields are valid"""
        cond = self.data.is_valid()
        cond &= self.augmentation.is_valid()
        cond &= self.training.is_valid()
        cond &= self.inference.is_valid()

        return cond


@dataclass
class DataConfig:
    """Class that defines the data arguments."""

    # pylint: disable=too-many-instance-attributes
    # the number is reasonable in our case.
    input_shape: List[int]
    images_train_dir: str

    annotations_train_file: str

    validation_split_size: float
    category_id_to_name: Dict[int, str]
    num_parallel_calls: int
    prefetch_value: int
    seed: int
    images_val_dir: Optional[str] = None
    annotations_val_file: Optional[str] = None

    def is_valid(self) -> bool:
        """ verify if the configuration fields are valid"""
        assert (
            self.input_shape[0] <= self.input_shape[1] and self.input_shape[2] == 3
        ), (
            "the height of input images must be less than or equal to their width "
            "and they need to have 3 color channels."
        )
        assert (
            self.validation_split_size <= 0.5
        ), "The validation split size must be less than or equal to 50 percent of the training set."

        return True


@dataclass
class AugmentationConfig:
    """Class that defines the data augmentation arguments."""

    rotation: float
    horizontal_flip: float
    random_brightness_contrast: float
    brightness_limit: Tuple[float, float]
    contrast_limit: Tuple[float, float]
    rotation_limit: Tuple[int, int]
    # Update: add the cropping args
    crop_erosion_rate: float
    crop_prob: float

    def is_valid(self) -> bool:
        """ verify if the configuration fields are valid"""
        assert (
            self.contrast_limit[1] <= 0.5
        ), "keep your contrast limit under 0.5 to avoid getting high contrasted images"
        assert (
            self.brightness_limit[1] <= 0.5
        ), "keep your bightness limit under 0.5 to avoid getting very bright images"
        assert (
            self.crop_erosion_rate <= 0.2
        ), "keep your erosion rate under 0.2 to avoid getting very small bboxes"

        return True


@dataclass
class TrainingConfig:
    """Class that defines the model's training arguments."""

    batch_size: int
    learning_rate: float
    num_batches: int
    model_config: str = ""
    model_name: str = ""
    checkpoint_path: str = ""
    output_model_path: str = ""
    logs_path: str = ""

    def is_valid(self) -> bool:
        """ verify if the configuration fields are valid"""
        assert (
            self.batch_size == 1 or self.batch_size % 2 == 0
        ), "the batch size need to be a power of 2 number"
        return True


@dataclass
class InferenceConfig:
    """Class that defines the inference arguments."""

    bbox_color: List[int]
    text_color: List[int]
    line_thickness: int
    figure_size: List[int]

    def is_valid(self) -> bool:
        """ verify if the configuration fields are valid"""
        assert (
            self.line_thickness <= 5
        ), "the line is too thick it must be lower than or equal to 5"
        return True


def get_config_from_yaml(config_path: str) -> "ExperimentConfig":
    """Read yaml file and returns corresponding config object."""
    with open(config_path, "r") as file:
        parameters = yaml.safe_load(file)

    # instantiate config object
    data_config = DataConfig(**parameters["data"])
    augmentation_config = AugmentationConfig(**parameters["augmentation"])
    training_config = TrainingConfig(**parameters["training"])
    inference_config = InferenceConfig(**parameters["inference"])
    ocr_config = OcrConfig.from_dict(parameters["ocr"])

    experiment_config = ExperimentConfig(
        data=data_config,
        augmentation=augmentation_config,
        training=training_config,
        inference=inference_config,
        ocr=ocr_config,
    )

    # checks if the obtained config is valid
    if not experiment_config.is_valid():
        raise ValueError("Tries to instantiate invalid experiment config.")

    return experiment_config
