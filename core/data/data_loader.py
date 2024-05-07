""" This module contains the data loading utils."""
# pylint: disable=no-name-in-module
# pylint: disable=too-many-locals

import json
import logging
import os
import random
from functools import partial
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from albumentations.augmentations.bbox_utils import normalize_bbox
from tensorflow.python.data.ops.dataset_ops import (
    Dataset,
    PrefetchDataset,
)

from core.data.data_augmentation import DataAugmentation
from core.utils.configs import ExperimentConfig


class DataLoader:
    """ This class defines the functions for data loading. """

    def __init__(
        self,
        config: ExperimentConfig,
        phase: str,
        image_dir_test: Optional[str] = None,
        annot_file_path_test: Optional[str] = None,
    ) -> None:
        self.config = config
        self.phase = phase
        self.image_dir_test = image_dir_test
        self.annot_file_path_test = annot_file_path_test

    def test_data_generator(self):
        """This function is used to generate the test data used do the inference.

        Yields:
            image [numpy.ndarray]: the generated image
            file_name[str]: the name of each image


        """
        random.seed(self.config.data.seed)

        image_directory = os.path.join(self.image_dir_test)
        images = os.listdir(image_directory)
        num_images = len(images)
        idx_list = list(range(num_images))

        for idx in idx_list:
            file_name = images[idx]
            image = cv2.imread(file_name)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            yield image, file_name

    def data_generator(self):
        """This function is used to generate images randomly with their annotations
         from a specific path.

        Yields:
            image [numpy.ndarray]: the generated image
            bboxes [list]: the bounding boxes describing each annotated object
            in the generated image
            category_ids[list]: the category ids of objects in each image.
            text_groundtruth [str]: the ground thruth text of the generated image
            file_name[str]: the name of each image

        """
        random.seed(self.config.data.seed)

        if self.phase == "train":
            annot_file_path = os.path.join(self.config.data.annotations_train_file)
            image_directory = os.path.join(self.config.data.images_train_dir)
            annot_file = json.load(open(annot_file_path))

            images = annot_file["images"]

            num_images = len(images)
            idx_list = list(range(num_images))
            # keep the train indexes only
            idx_list = idx_list[
                int(self.config.data.validation_split_size * num_images) :
            ]
            random.shuffle(idx_list)
        elif self.phase == "val":
            # Take the validation data as split from the training data

            if (
                self.config.data.images_val_dir == None
                and self.config.data.annotations_val_file == None
            ):

                annot_file_path = os.path.join(self.config.data.annotations_train_file)
                image_directory = os.path.join(self.config.data.images_train_dir)
                annot_file = json.load(open(annot_file_path))

                images = annot_file["images"]

                num_images = len(images)

                idx_list = list(range(num_images))

                # Keep the validation indexes only
                idx_list = idx_list[
                    : int(self.config.data.validation_split_size * num_images)
                ]

            # Take the validation data from the config file

            else:
                annot_file_path = os.path.join(self.config.data.annotations_val_file)
                image_directory = os.path.join(self.config.data.images_val_dir)
                annot_file = json.load(open(annot_file_path))

                images = annot_file["images"]

                num_images = len(images)

                idx_list = list(range(num_images))

            # Keep the validation indexes only
            idx_list = idx_list[
                : int(self.config.data.validation_split_size * num_images)
            ]

        elif self.phase == "eval":
            annot_file_path = os.path.join(self.annot_file_path_test)
            image_directory = os.path.join(self.image_dir_test)
            annot_file = json.load(open(annot_file_path))

            images = annot_file["images"]

            num_images = len(images)
            idx_list = list(range(num_images))
        else:
            raise RuntimeError("Please specify a valid phase: train,val,eval")

        for idx in idx_list:
            file_name = images[idx]["file_name"]
            image_path = os.path.join(file_name)
            image = cv2.imread(image_path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height = images[idx]["height"]
            image_width = images[idx]["width"]

            bboxes = images[idx]["bbox"]
            normalized_bboxes = [
                list(normalize_bbox(bbox, image_height, image_width)) for bbox in bboxes
            ]
            category_ids = images[idx]["category_id"]
            # We return an empty string for training as we do not train the ocr.
            if self.phase == "train":
                text_groundtruth = ""
            else:
                text_groundtruth = images[idx]["text_groundtruth"]
            text_groundtruth_list = [text_groundtruth] * len(bboxes)
            yield image, normalized_bboxes, category_ids, text_groundtruth_list, file_name

    def preprocess_data(self, dataset: Dataset) -> PrefetchDataset:
        """This Function is used to preprocess the input dataset.

        Args:
            dataset: the image dataset

        Returns:
            preprocessed_dataset: the augmented and preprocessed dataset

        """
        data_augmentation = DataAugmentation(config=self.config, phase=self.phase)

        preprocessed_dataset = dataset.map(
            partial(data_augmentation.process_data),
            num_parallel_calls=self.config.data.num_parallel_calls,
        ).prefetch(self.config.data.prefetch_value)

        return preprocessed_dataset


def get_dataset(
    config: ExperimentConfig,
    phase: str,
    image_dir: Optional[str] = None,
    annot_file_path: Optional[str] = None,
) -> Dataset:
    """This Function is used to generate the final augmented and batched
    training/validation datasets.


    Args:
        config: the configuration file.
        phase: the data loading phase.
        image_dir: the directory containing the test data.
        annot_file_path: the path of the annotation file for evaluation.
    Returns:
        batched_dataset: the augmented and batched training dataset

    """
    data_loader = DataLoader(
        config=config,
        phase=phase,
        image_dir_test=image_dir,
        annot_file_path_test=annot_file_path,
    )

    logging.info("Phase: %s ", phase)
    if phase == "inference":

        dataset = tf.data.Dataset.from_generator(
            generator=data_loader.test_data_generator,
            output_types=(tf.int8),
        )
        batched_dataset = dataset.batch(batch_size=config.training.batch_size).prefetch(
            config.data.prefetch_value
        )

    else:

        dataset = tf.data.Dataset.from_generator(
            generator=data_loader.data_generator,
            output_types=(tf.int8),
        )

        preprocessed_dataset = data_loader.preprocess_data(dataset=dataset)
        batched_dataset = preprocessed_dataset.padded_batch(
            batch_size=config.training.batch_size,
            padded_shapes=([None, None, None], [None, None], [None, None], [None], []),
            drop_remainder=False,
        ).prefetch(config.data.prefetch_value)

    return batched_dataset
