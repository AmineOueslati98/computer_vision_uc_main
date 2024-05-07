""" This module contains the visualization utils."""
# pylint: disable=no-name-in-module

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset

from core.utils.configs import ExperimentConfig


class Visualization:
    """ This class defines the visualization utils."""

    def __init__(self, config: ExperimentConfig, phase: str) -> None:
        self.config = config
        self.phase = phase

    def view_image(self, dataset: Dataset, block: bool = True) -> None:
        """This Function is used to view one batch of the constructed dataset.

        Args:
            dataset: the batched dataset
            block: this argument keeps the visualization plotted.



        """
        if self.phase == "inference":
            image, _ = next(iter(dataset))

        elif self.phase == "train" or self.phase == "val" or self.phase == "eval":

            image, _, _, _, _ = next(iter(dataset))

        fig = plt.figure(figsize=self.config.inference.figure_size)
        image = image.numpy()

        for i in range(self.config.training.batch_size):

            axis = fig.add_subplot(
                1, self.config.training.batch_size, i + 1, xticks=[], yticks=[]
            )

            axis.imshow(image[i].astype("uint8"))
            axis.set_title(f"image {i+1} ")
        plt.show(block=block)

    def visualize_bbox(
        self, image: np.ndarray, bbox: np.ndarray, label: str
    ) -> np.ndarray:
        """This Function is used to visualize a single bounding box on the image.

        Args:
        image[numpy.ndarray]: the image to view
        bbox[numpy.ndarray]: a single bounding box coordinates
        label[string]: the name of the recognized class


        Returns:
            image: returns the image with the bounding box arround an object
        """
        height, width, _ = image.shape
        y_min, x_min, y_max, x_max = bbox
        x_min, y_min, x_max, y_max = (
            int(x_min * width),
            int(y_min * height),
            int(x_max * width),
            int(y_max * height),
        )

        cv2.rectangle(
            image,
            (x_min, y_min),
            (x_max, y_max),
            color=self.config.inference.bbox_color,
            thickness=self.config.inference.line_thickness,
        )

        ((text_width, text_height), _) = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
        )
        cv2.rectangle(
            image,
            (x_min, y_min - int(1.3 * text_height)),
            (x_min + text_width, y_min),
            self.config.inference.bbox_color,
            -1,
        )
        cv2.putText(
            image,
            text=label,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=self.config.inference.text_color,
            lineType=cv2.LINE_AA,
        )
        return image

    def visualize_annotated_objects(
        self, dataset: Dataset, block: bool = True
    ) -> None:
        """This Function is used to visualize the overlapping  bounding boxes on the image

        Args:
            dataset: the batched dataset
            block: this argument keeps the visualization plotted.
        """

        image, bboxes, category_ids, _, _ = next(iter(dataset))
        bboxes = bboxes.numpy()

        image = image.numpy()
        encoded_category_ids = category_ids.numpy()

        fig = plt.figure(figsize=self.config.inference.figure_size)
        print(f"This is a batch of {self.config.training.batch_size} photos")
        for i in range(self.config.training.batch_size):
            label_indexes = tf.argmax(encoded_category_ids[i], axis=1).numpy()
            # Filter the invalid bounding boxes (containing zeros)

            print(
                f"image {i+1} contains {np.count_nonzero(bboxes[i])//4} valid object(s):"
            )

            for bbox, index in zip(bboxes[i], label_indexes):
                label = self.config.data.category_id_to_name[index]
                img = self.visualize_bbox(image[i], bbox, label)
                if np.count_nonzero(bbox) > 0:

                    print(f"bbox: {bbox}")
                    print(f"label: {label}")

                else:
                    pass

            axis = fig.add_subplot(
                1, self.config.training.batch_size, i + 1, xticks=[], yticks=[]
            )

            axis.imshow(img.astype("uint8"))
        plt.show(block=block)
