"""OCR class"""
import os
import re
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from paddleocr import PaddleOCR
import easyocr

from core.ocr.config import OcrConfig
from core.ocr.image_preprocessing import create_preprocessed_image_batch
from core.ocr.text_postprocessor import SignOcrPostprocessor, TextPostprocessor


class OcrEngine:
    """Class Wrapping multiple OCR Engines : PaddleOCR"""

    def __init__(self, config) -> None:
        """Initialize the class by loading the chosen OCR"""
        self.config = config
        self.model = None
        self.paddle = False
        self.easy = False

        if "paddle" in self.config.model.engine.lower():
            self.paddle = True
            kwrgs = {
                "lang": "en",
                "gpu_mem": self.config.model.gpu_memory,
                "rec_image_shape": self.config.model.rec_image_shape,
                "use_space_char": self.config.model.use_space_char,
                "drop_score": self.config.model.drop_score,
                "max_text_length": self.config.model.max_text_length,
            }
            if (
                self.config.model.paddle_pretrained is not None
                and len(self.config.model.paddle_pretrained) > 0
            ):
                kwrgs["rec_model_dir"] = self.config.model.paddle_pretrained
                kwrgs["rec_char_dict_path"] = str(
                    os.path.join(self.config.model.paddle_pretrained, "dict.txt")
                )
            self.model = PaddleOCR(**kwrgs)

        # Update : add easyOCR support
        elif  "easy" in self.config.model.engine.lower():
            self.easy = True
            self.model = easyocr.Reader(["en"])
        
        else:
            raise RuntimeError("Please specify a valid OCR engine: paddle or easy")


    def ocr(self, image_list, det=False):
        """Performing OCR.
        args:
            image_list: list of images of shape [h, w, 3]
        returns:
            detected_text: list of text returned for each images
        """
        if self.paddle:
            result = self.model.ocr(image_list, det)
            return result
        
        # Update : add easyOCR support
        elif self.easy:
            result = self.model.recognize(image_list[0], allowlist='0123456789km.,')
            return result
        else:
            raise RuntimeError("Initialization error: Please specify a valid OCR engine: paddle or easy")


class OcrModel:
    """OCR model built on paddle_ocr."""

    def __init__(self, ocr_config: OcrConfig):
        """Constructor.
        Args:
        ocr_config: config of ocr
        """
        self.config = ocr_config
        self.model = OcrEngine(ocr_config)
        self.postprocessing: TextPostprocessor = SignOcrPostprocessor(
            ocr_config.postprocessing
        )

    def apply_ocr(
        self, image_list: List[np.ndarray], detect_box: bool = False
    ) -> List[str]:
        """
        Apply ocr on a list of images
        args:
            image_list: list of images of shape [h, w, 3]
        returns:
            detected_text: list of text returned for each images
        """
        if detect_box:
            # detction works on a single image
            results = []
            for image in image_list:
                ocr_result = self.model.ocr(image, det=True)
                if ocr_result:
                    max_detected_score = 0
                    best_detected_text = ""
                    for detected_box in ocr_result:
                        if detected_box[1][1] > max_detected_score:
                            detected_text = detected_box[1][0]
                            if self.config.model.post_process:
                                detected_text = self.postprocessing(detected_text)
                            if detected_text:
                                max_detected_score = detected_box[1][1]
                                best_detected_text = detected_text

                    results.append((best_detected_text, max_detected_score))
                else:
                    results.append(("", 0))
        else:
            results = self.model.ocr(image_list, det=False)

        # Update : add easyOCR support
        if self.config.model.engine=='paddle':
            results = [result[0] for result in results]
            
        elif self.config.model.engine=='easy':
            results = [result[1] for result in results]
        
        if self.config.model.post_process:
            return [self.postprocessing(result) for result in results]

        return [result for result in results]


    def __call__(
        self,
        images: np.ndarray,
        images_info_list: Optional[List[Dict[str, Any]]],
        bounding_boxes: Optional[np.ndarray],
        class_names: Optional[np.ndarray],
    ):
        """
        Preprocess images and apply ocr on them.
        args:
            images: batch of images of shape [n, h, w, 3]
            images_info_list: list containing a dict {"index": image_index,
                "class_name": str, "bbox": (ymin, xmin, ymax, xmax)}. Either this variable or images_bounding_boxes and images_category_ids should be not None.
            images_bounding_boxes: array of shape [n, num_bboxes, 4] containing the bounding boxes in this format [ymin, xmin, ymax, xmax].
            images_class_names: array of shape [n, num_bboxes] containing the class_name of each bbox.
        returns:
            detected_text: list of text returned for each bbox
        """
        # when detect_box is True we dont need cropping inside the detected sign
        if self.config.model.detect_box:
            self.config.preprocessing.crop_box_coordinate = [0, 0, 1, 1]

        preprocessed_images = create_preprocessed_image_batch(
            images,
            images_info_list,
            bounding_boxes,
            class_names,
            self.config.preprocessing,
        )

        # paddle_ocr requires BGR images
        preprocessed_images = [
            cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
            for processed_image in preprocessed_images
        ]
        return self.apply_ocr(preprocessed_images, self.config.model.detect_box)


if __name__ == "__main__":
    import logging
    import os
    from difflib import SequenceMatcher
    from xml.etree import ElementTree

    from core.ocr.config import (
        OcrModelConfig,
        OcrPreprocessingConfig,
        SignOcrPostprocessorConfig,
    )

    def score_image_batch(
        image_folder_path, annotation_folder_path, config, detect_box=False
    ):
        score = 1
        images = []
        images_bounding_boxes = []
        ground_truth_list = []
        number_of_objects = 0
        for image_num, filename in enumerate(os.listdir(image_folder_path)):
            image_path = os.path.join(image_folder_path, filename)
            images.append(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
            xml_file_name = filename[: filename.find(".")] + ".xml"
            annotation_file_path = os.path.join(annotation_folder_path, xml_file_name)
            annotation_file = ElementTree.parse(annotation_file_path).getroot()
            ground_truth = annotation_file.find("length").text
            score_sum = 0
            for detected_object in annotation_file.iter("object"):
                ground_truth_list.append(ground_truth)
                bbox_xml = detected_object.find("bndbox")
                bbox = [
                    int(bbox_xml.find("ymin").text),
                    int(bbox_xml.find("xmin").text),
                    int(bbox_xml.find("ymax").text),
                    int(bbox_xml.find("xmax").text),
                ]
                bbox_dict = {"index": image_num, "bbox": bbox}
                images_bounding_boxes.append(bbox_dict)
                number_of_objects += 1

        if number_of_objects:
            score_sum = 0
            ocr = OcrModel(config)
            result = ocr(np.array(images), images_bounding_boxes)
            for ground_truth, ocr_result in zip(ground_truth_list, result):
                score = SequenceMatcher(None, ground_truth, ocr_result).ratio()
                logging.info(
                    "ground truth: %s  |||  detected word: %s ||| score: %s",
                    ground_truth,
                    ocr_result,
                    score,
                )
                score_sum += score
            logging.info("The total score is: %f", score_sum / number_of_objects)
        else:
            logging.warning("Empty image folder")
        return score

    config = OcrConfig(
        model=OcrModelConfig(),
        preprocessing=OcrPreprocessingConfig(
            cropped_part_size=[100, 200], apply_super_resolution_gan=True
        ),
        postprocessing=SignOcrPostprocessorConfig(),
    )

    score_image_batch(
        "dataset-1280/train/best_images",
        "dataset-1280/train/annotations_xml",
        config,
        detect_box=True,
    )
