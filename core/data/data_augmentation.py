""" This module contains the data preprocessing and augmentation utils."""

from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from albumentations import (
    BboxParams,
    Compose,
    HorizontalFlip,
    RandomBrightnessContrast,
    RandomSizedBBoxSafeCrop,
    Rotate,
)

from core.utils.configs import ExperimentConfig


class DataAugmentation:
    """ This class defines the data preprocessing and augmentation functions."""

    def __init__(self, config: ExperimentConfig, phase: str) -> None:
        self.config = config
        self.phase = phase

    def preprocess_val_test_data(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        category_ids: List[float],
        text_groundtruth: List[str],
        file_name: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """This Function is used to preprocess the validation and test data.

        Args:
        image[numpy.ndarray]: the input image
        bboxes[list]: the normalized bounding boxes coordinates
        category_ids[list]: a list containing the id of each annotated object
        text_groundtruth[list]: a list of text groundtruth
        file_name[str]: the name of each image.

        Returns:
            resized_image[numpy.ndarray]: the resized image
            bboxes[numpy.ndarray]: the normalized bounding boxes coordinates
            category_ids[numpy.ndarray]: a list containing the id of each annotated object
            text_groundtruth[list]: a list of text groundtruth
            file_name[str]: the name of each image.


        """
        # The height, width and number of channels of input images
        height, width, _ = self.config.data.input_shape
        resized_image = cv2.resize(
            image, dsize=(height, width), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        # Putting the bounding boxes into [ymin,xmin,ymax,xmax] format
        new_bboxes = [[bbox[1], bbox[0], bbox[3], bbox[2]] for bbox in bboxes]

        bboxes = np.array(new_bboxes, np.float32)
        category_ids = np.array(category_ids, dtype=np.float32)
        text_groundtruth = np.array(text_groundtruth, dtype=np.str)
        file_name = np.array(file_name, dtype=np.str)

        return resized_image, bboxes, category_ids, text_groundtruth, file_name

    def augment_train_data(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        category_ids: List[float],
        text_groundtruth: List[str],
        file_name: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        """This Function is used to apply the data augmentations to the training data.

        Args:
        image[numpy.ndarray]: the input image
        bboxes[list]: the normalized bounding boxes coordinates
        category_ids[list]: a list containing the id of each annotated object
        text_groundtruth[list]: a list of text groundtruth
        file_name[str]: the name of each image.

        Returns:
            aug_img[numpy.ndarray]: the augmented image
            aug_bboxes[numpy.ndarray]: the augmented bounding boxes set in
                [ymin,xmin,ymax,xmax] format.
            aug_ids[numpy.ndarray]: an array containing the augmented id of each annotated object
            text_groundtruth[np.ndarray]: the text ground truth
            file_name[str]: the name of each image.
        """
        # The height, width and number of channels of input images used for resizing
        height, width, _ = self.config.data.input_shape
        # Defining the data augmentation transforms
        transforms = Compose(
            [
                HorizontalFlip(p=self.config.augmentation.horizontal_flip),
                Rotate(
                    limit=(self.config.augmentation.rotation_limit),
                    p=self.config.augmentation.rotation,
                ),
                RandomBrightnessContrast(
                    brightness_limit=self.config.augmentation.brightness_limit,
                    contrast_limit=self.config.augmentation.contrast_limit,
                    p=self.config.augmentation.random_brightness_contrast,
                ),
            ],
            bbox_params=BboxParams(
                format="albumentations", label_fields=["category_ids"]
            ),
        )

        aug_data = transforms(image=image, bboxes=bboxes, category_ids=category_ids)

        aug_img = aug_data["image"].astype(np.float32)
        aug_bboxes = aug_data["bboxes"]
        # Putting the bounding boxes into [ymin,xmin,ymax,xmax] format
        new_bboxes = [[bbox[1], bbox[0], bbox[3], bbox[2]] for bbox in aug_bboxes]

        aug_bboxes = np.array(new_bboxes, np.float32)
        aug_ids = aug_data["category_ids"]

        aug_ids = np.array(aug_ids, dtype=np.float32)
        text_groundtruth = np.array(text_groundtruth, dtype=np.str)
        file_name = np.array(file_name, dtype=np.str)

        return aug_img, aug_bboxes, aug_ids, text_groundtruth, file_name

    def process_data(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        category_ids: List[float],
        text_groundtruth: List[str],
        file_name: str,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

        """This Function is used to wrap the augmented images into tensorflow ops.

        Args:
        image[numpy.ndarray]: the input image
        bboxes[list]: the overlapping the bounding boxes coordinates
        category_ids[list]: a list containing the id of each annotated object
        text_groundtruth[list]: a list containing the text ground truth
        file_name[str]: the name of each image.


        Returns:
            aug_img[tf.Tensor]: a tensor containing the augmented image
            aug_bboxes[tf.Tensor]: a tensor containing the augmented bounding boxes
            aug_ids[tf.Tensor]: a tensor containing the id of each annotated object
            text_groundtruth[tf.Tensor]: a tensor containing the text ground truth
            file_name[tf.Tensor]: the name of each image.

        """

        if self.phase == "train":

            (
                aug_img,
                aug_bboxes,
                aug_ids,
                text_groundtruth,
                file_name,
            ) = tf.numpy_function(
                func=self.augment_train_data,
                inp=[image, bboxes, category_ids, text_groundtruth, file_name],
                Tout=(tf.float32, tf.float32, tf.float32, tf.string, tf.string),
            )

        elif self.phase == "val" or self.phase == "eval":
            (
                aug_img,
                aug_bboxes,
                aug_ids,
                text_groundtruth,
                file_name,
            ) = tf.numpy_function(
                func=self.preprocess_val_test_data,
                inp=[image, bboxes, category_ids, text_groundtruth, file_name],
                Tout=(tf.float32, tf.float32, tf.float32, tf.string, tf.string),
            )
        num_classes = len(self.config.data.category_id_to_name)
        # Reshape augmented bounding boxes and category_ids
        if tf.equal(tf.size(aug_bboxes), 0):
            aug_bboxes = tf.zeros([1, 4])

        if tf.equal(tf.size(aug_ids), 0):
            aug_ids = tf.zeros([1, num_classes])
        # Encode the ids of the valid objects after image augmentation
        else:

            aug_ids = tf.one_hot(
                indices=tf.cast(aug_ids, tf.int64),
                depth=num_classes,
                on_value=None,
                off_value=None,
            )

        return aug_img, aug_bboxes, aug_ids, text_groundtruth, file_name
