"""Functions to preprocess images before running OCR on them."""
import concurrent.futures
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ppgan.apps import RealSRPredictor

from core.ocr.config import OcrPreprocessingConfig, ThresholdingMode


def invert(image: np.ndarray) -> np.ndarray:
    """Invert the colours of an image.

    args:
        image: input image
    returns:
        inverted_image: result image
    """
    return 255 - image


# get grayscale image
def get_grayscale(image: np.ndarray) -> np.ndarray:
    """Change the image from BGR to Gray.

    args:
        image: input image
    returns:
        gray_image: grayscale image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# noise removal
def remove_noise(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Smooth the image using the median filter.

    args:
        image: input image
        kernel_size: filter kernel size. Needs to be an odd number greater
         than 1.
    returns:
        blurred_image: result image
    """
    return cv2.medianBlur(image, kernel_size)


# thresholding
def thresholding(image: np.ndarray, ocr_config: OcrPreprocessingConfig) -> np.ndarray:
    """Apply thershold on image.

    args:
        image: input image
        ocr_config: instance of OcrPreprocessingConfig class
    returns:
        thresholded_image: result image
    """
    if ocr_config.threshold_mode == ThresholdingMode.Binary:
        return cv2.threshold(image, ocr_config.threshold, 255, cv2.THRESH_BINARY)[1]
    if ocr_config.threshold_mode == ThresholdingMode.Otsu:
        return cv2.threshold(
            image, ocr_config.threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
    if ocr_config.threshold_mode == ThresholdingMode.AdaptiveGaussian:
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ocr_config.adaptive_threshold_bloc_size,
            ocr_config.adaptive_threshold_constant,
        )
    return image


# dilation
def dilate(
    image: np.ndarray, kernel_size: List[int], number_of_iterations: int
) -> np.ndarray:
    """Dilate the source image.

    args:
        image: input image
        kernel_size: dilation kernel size
        number_of_iterations: number of time the dilation is applied
    returns:
        dilated_image: result image
    """
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=number_of_iterations)


# erosion
def erode(
    image: np.ndarray, kernel_size: List[int], number_of_iterations: int
) -> np.ndarray:
    """Erode the source image.

    args:
        image: input image
        kernel_size: erosion kernel size
        number_of_iterations: number of time the erosion is applied
    returns:
        eroded_image: result image
    """
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=number_of_iterations)


# canny edge detection
def canny(image: np.ndarray, threshold1: int = 50, threshold2: int = 100) -> np.ndarray:
    """Find edges in the input image using the Canny algorithm.

    args:
        image: input image
        thershold1: first threshold for the hysteresis procedure
        threshold2: second threshold for the hysteresis procedure
    returns:
        edged_image: result image
    """
    return cv2.Canny(image, threshold1, threshold2, apertureSize=3)


def change_brightness_contrast(
    image: np.ndarray, brightness: int = 160, contrast: int = 100
):
    """Change image brightness and contrast.

    args:
        image: input image
        brightness: requested brightness preferably value between [-127, 127]
        contrast: requested contrast value preferably between [-127, 127]
    returns:
        result_image: image with the new brightness and contrast
    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        result_image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    else:
        result_image = image.copy()

    if contrast != 0:
        alpha_c = 131 * (contrast + 127) / (127 * (131 - contrast))
        gamma_c = 127 * (1 - alpha_c)

        result_image = cv2.addWeighted(result_image, alpha_c, result_image, 0, gamma_c)

    return result_image


def deskew(
    image: np.ndarray, h_range: List[int], s_range: List[int], v_range: List[int]
) -> np.ndarray:
    """Use canny edges to determine important part corner points and change perspective
     to make it straight.

    args:
        image: input image
    returns:
        image: deskewed image
    """
    # pylint: disable=too-many-locals
    # the number is reasonable in our case.
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_not_in_range = (hsv_image[:, :, 0] < h_range[0]) | (
        hsv_image[:, :, 0] > h_range[1]
    )
    s_not_in_range = (hsv_image[:, :, 1] < s_range[0]) | (
        hsv_image[:, :, 1] > s_range[1]
    )
    v_not_in_range = (hsv_image[:, :, 2] < v_range[0]) | (
        hsv_image[:, :, 2] > v_range[1]
    )
    hsv_image[h_not_in_range | s_not_in_range | v_not_in_range] = [0, 0, 0]

    hsv_image = remove_noise(hsv_image, 5)
    hsv_image = hsv_image[:, :, 1]
    hsv_image = canny(hsv_image)
    contour_points_indices = np.argwhere(hsv_image > 0)

    if contour_points_indices.size:
        ind = np.argmin(np.sum(contour_points_indices, axis=1))
        top_left_contour_indice = (contour_points_indices[ind])[::-1]
        ind = np.argmin(
            np.sum(
                np.abs(
                    contour_points_indices - np.array([image.shape[0], image.shape[1]])
                ),
                axis=1,
            )
        )
        bottom_right_contour_indice = (contour_points_indices[ind])[::-1]
        ind = np.argmin(
            np.sum(
                np.abs(contour_points_indices - np.array([0, image.shape[1]])), axis=1
            )
        )
        top_right_contour_point = (contour_points_indices[ind])[::-1]
        ind = np.argmin(
            np.sum(
                np.abs(contour_points_indices - np.array([image.shape[0], 0])), axis=1
            )
        )
        bottom_left_contour_point = (contour_points_indices[ind])[::-1]
    else:
        top_left_contour_indice = [0, 0]
        bottom_right_contour_indice = [image.shape[0] - 1, image.shape[1] - 1]
        top_right_contour_point = [0, image.shape[1] - 1]
        bottom_left_contour_point = [image.shape[0] - 1, 0]

    width_1 = np.sqrt(
        ((top_left_contour_indice[0] - top_right_contour_point[0]) ** 2)
        + ((top_left_contour_indice[1] - top_right_contour_point[1]) ** 2)
    )
    width_2 = np.sqrt(
        ((bottom_left_contour_point[0] - bottom_right_contour_indice[0]) ** 2)
        + ((bottom_left_contour_point[1] - bottom_right_contour_indice[1]) ** 2)
    )
    max_width = max(int(width_1), int(width_2))

    height_1 = np.sqrt(
        ((top_left_contour_indice[0] - bottom_left_contour_point[0]) ** 2)
        + ((top_left_contour_indice[1] - bottom_left_contour_point[1]) ** 2)
    )
    height_2 = np.sqrt(
        ((bottom_right_contour_indice[0] - top_right_contour_point[0]) ** 2)
        + ((bottom_right_contour_indice[1] - top_right_contour_point[1]) ** 2)
    )
    max_height = max(int(height_1), int(height_2))
    input_pts = np.float32(
        [
            top_left_contour_indice,
            bottom_left_contour_point,
            bottom_right_contour_indice,
            top_right_contour_point,
        ]
    )
    output_pts = np.float32(
        [
            [0, 0],
            [0, max_height - 1],
            [max_width - 1, max_height - 1],
            [max_width - 1, 0],
        ]
    )
    # Compute the perspective transform M
    perspective_transform_matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
    image = cv2.warpPerspective(
        image,
        perspective_transform_matrix,
        (max_width, max_height),
        flags=cv2.INTER_LINEAR,
    )

    return image


def crop_images_from_dict(
    images: np.ndarray, bounding_boxes: List[Dict[str, Any]], pad_images: bool
):
    result_images = []
    result_bounding_boxes = []
    for bounding_box_info in bounding_boxes:
        result_images.append(images[int(bounding_box_info["index"]), ...])
        result_bounding_boxes.append(bounding_box_info["bbox"])
    return crop_images(
        np.array(result_images), np.array(result_bounding_boxes), pad_images
    )


def crop_images(
    images: np.ndarray, bounding_boxes: np.ndarray, pad_images: bool
) -> Union[np.ndarray, List[np.ndarray]]:
    """Crop a batch of images [n, h, w, c] using batch of bounding boxes [n, 4].
    
    args:
        images: batch of images [n, h, w, c]
        bounding_boxes: batch of bounding boxes [n, 4] with each bounding box [ymin, xmin, ymax, xmax]
    returns:
        cropped_images: batch of cropped and padded images if pad images is true otherwise a list of cropped images.
        original_size_list: size of the cropped images before the padding
    """
    cropped_images = []
    max_height = 0
    max_width = 0
    for image_num in range(images.shape[0]):
        if (
            bounding_boxes[image_num, 0] < bounding_boxes[image_num, 2]
            and bounding_boxes[image_num, 1] < bounding_boxes[image_num, 3]
        ):
            cropped_image = images[
                image_num,
                bounding_boxes[image_num, 0] : bounding_boxes[image_num, 2],
                bounding_boxes[image_num, 1] : bounding_boxes[image_num, 3],
            ]
        else:
            # corrupted bbox. The image will be returned as it is.
            cropped_image = images[image_num, ...]
        height, width, _ = cropped_image.shape
        cropped_images.append(cropped_image)
        if height > max_height:
            max_height = height
        if width > max_width:
            max_width = width
    if pad_images:

        padded_image_list = []
        original_size_list = []
        for image in cropped_images:
            height, width, _ = image.shape
            original_size_list.append((height, width))
            added_height = max_height - height
            added_width = max_width - width
            if added_height or added_width:
                image = cv2.copyMakeBorder(
                    image,
                    0,
                    added_height,
                    0,
                    added_width,
                    cv2.BORDER_CONSTANT,
                    (255, 255, 255),
                )
            padded_image_list.append(image)

    else:
        original_size_list = [
            (cropped_image.shape[0], cropped_image.shape[1])
            for cropped_image in cropped_images
        ]
        return cropped_images, original_size_list
    return np.array(padded_image_list, dtype=np.uint8), original_size_list


def run_super_resolution_gan(super_resolution, image_batch, original_size_list):
    super_resolution_images = super_resolution.run_multiple_images(image_batch)
    result_images = []
    for im_num in range(super_resolution_images.shape[0]):
        im = super_resolution_images[
            im_num,
            0 : 4 * original_size_list[im_num][0],
            0 : 4 * original_size_list[im_num][1],
            :,
        ]
        result_images.append(im)
    return result_images


def crop_image(image: np.ndarray, crop_box_coordinate: List[float]) -> np.ndarray:
    """Crop an image using normalized coordinate.

    args:
        image: input image
        crop_box_coordinate: normalized coordinate as follow [left, upper, right, lower]
    returns:
        cropped_part: cropped part of the image
    """
    height, width, _ = image.shape
    return image[
        int(crop_box_coordinate[1] * height) : int(crop_box_coordinate[3] * height),
        int(crop_box_coordinate[0] * width) : int(crop_box_coordinate[2] * width),
    ]


# pylint: disable=bare-except
def preprocess_image(
    image: np.ndarray, class_name: str, ocr_config: OcrPreprocessingConfig
) -> np.ndarray:
    """Apply all the configured preprocessing technique on the image.

    args:
        image: input image in RGB format
        class_name: the class name of the image
        ocr_config: instance of OcrPreprocessingConfig
    returns:
        image: preprocessed image
    """
    # make a copy of the image
    image = image.astype(np.uint8)
    cropped_image_copy = image.copy()

    # check if the image not empty
    if image.size != 0:
        try:
            if ocr_config.apply_brightness_contrast:
                # change brightness and contrast relatively to the image current brightness and contrast
                hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                image_brightness = hsv_image[:, :, 2].mean()
                img_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image_contrast = img_grey.std()

                image = change_brightness_contrast(
                    image,
                    ocr_config.brightness - image_brightness,
                    ocr_config.contrast - image_contrast,
                )

            if ocr_config.apply_deskewing:
                image = deskew(
                    image,
                    ocr_config.h_range_for_deskewing[class_name],
                    ocr_config.s_range_for_deskewing[class_name],
                    ocr_config.v_range_for_deskewing[class_name],
                )

            # crop the useless parts inside the bounding box
            image = crop_image(image, ocr_config.crop_box_coordinate)

            if ocr_config.cropped_part_size[0] and ocr_config.cropped_part_size[1]:
                image = cv2.resize(
                    image,
                    dsize=tuple(ocr_config.cropped_part_size),
                    interpolation=cv2.INTER_CUBIC,
                )

            image = get_grayscale(image)
            if ocr_config.apply_inverting:
                image = invert(image)

            if ocr_config.apply_histogram_equilization:
                clahe = cv2.createCLAHE(
                    clipLimit=ocr_config.clip_limit,
                    tileGridSize=tuple(ocr_config.tile_grid_size),
                )
                image = clahe.apply(image)

            if ocr_config.apply_adding_border:
                image = cv2.copyMakeBorder(
                    image,
                    ocr_config.border_box[1],
                    ocr_config.border_box[3],
                    ocr_config.border_box[0],
                    ocr_config.border_box[2],
                    cv2.BORDER_CONSTANT,
                    value=[255],
                )

            if ocr_config.apply_median_blur:
                image = remove_noise(image, ocr_config.median_blur_kernel_size)

            if ocr_config.apply_erosion:
                image = erode(
                    image,
                    ocr_config.erosion_kernel_size,
                    ocr_config.erosion_number_of_iterations,
                )

            if ocr_config.apply_dilation:
                image = dilate(
                    image,
                    ocr_config.dilation_kernel_size,
                    ocr_config.dilation_number_of_iterations,
                )

            if ocr_config.apply_thresholding:
                image = thresholding(image, ocr_config)
        except:
            logging.warning("Image preprocessing failed!")
            return get_grayscale(cropped_image_copy)
    else:
        return get_grayscale(cropped_image_copy)
    return image


# Preprocess a single image
def create_preprocessed_image(
    image_arg: Tuple[int, np.ndarray, str, OcrPreprocessingConfig]
) -> np.array:
    """Apply preprocess image on an image.

    args:
        image_arg: tuple containing (index, input image, class_name, ocr config)
    returns:
        preprocessed_image: preprocessed image"""
    _, image, class_name, ocr_config = image_arg
    preprocessed_image = preprocess_image(image, class_name, ocr_config)
    return preprocessed_image


def create_preprocessed_image_batch(
    images: np.ndarray,
    images_info_list: Optional[List[Dict[str, Any]]],
    images_bounding_boxes: Optional[np.ndarray],
    images_class_names: Optional[np.ndarray],
    ocr_config: OcrPreprocessingConfig,
) -> List[np.array]:
    """Preprocess multiple images concurrently.

    args:
        images: list of input images
        images_info_list: list containing a dict {"index": image_index,
         "class_name": str, "bbox": (ymin, xmin, ymax, xmax)}. Either this variable or images_bounding_boxes and images_category_ids should be not None.
        images_bounding_boxes: array of shape [n, num_bboxes, 4] containing the bounding boxes in this format [ymin, xmin, ymax, xmax].
        images_class_names: array of shape [n, num_bboxes] containing the class_name of each bbox.
        ocr config: an instance of OcrPreprocessingConfig
    returns:
        preprocessed_images: list of preprocessed images
    """
    # The padding is added only when we need to run the super resolution gan
    add_padding = ocr_config.apply_super_resolution_gan
    # one of images_info_list or (images_bounding_boxes and images_category_ids) is not None
    assert (images_info_list is not None) or (
        images_bounding_boxes is not None and images_class_names is not None
    )
    if images_info_list != None:
        assert isinstance(images_info_list, list)
        image_class_list = [image_info["class_name"] for image_info in images_info_list]
        images, original_size_list = crop_images_from_dict(
            images, images_info_list, add_padding
        )
    else:
        assert (
            isinstance(images_bounding_boxes, np.ndarray)
            and isinstance(images_class_names, np.ndarray)
            and images_class_names.shape[:2] == images_bounding_boxes.shape[:2]
            and images.shape[0] == images_bounding_boxes.shape[0]
            and images_bounding_boxes.shape[2] == 4
        )
        images_list = []
        bboxes_list = []
        image_class_list = []
        for image_num in range(images.shape[0]):
            for box_num in range(images_bounding_boxes.shape[1]):
                images_list.append(images[image_num, ...])
                bboxes_list.append(images_bounding_boxes[image_num, box_num, ...])
                image_class_list.append(images_class_names[image_num, box_num])
        images = np.array(images_list)
        images_bounding_boxes = np.array(bboxes_list)
        images, original_size_list = crop_images(
            images, images_bounding_boxes, add_padding
        )

    if ocr_config.apply_super_resolution_gan:
        super_resolution = RealSRPredictor(output="")

        total_number_of_images = images.shape[0]
        srg_number_of_runs = math.ceil(
            total_number_of_images / ocr_config.super_resolution_batch_size
        )
        srg_images = []
        for iteration in range(srg_number_of_runs):
            srg_images += run_super_resolution_gan(
                super_resolution,
                images[
                    iteration
                    * ocr_config.super_resolution_batch_size : (iteration + 1)
                    * ocr_config.super_resolution_batch_size,
                    ...,
                ],
                original_size_list,
            )
    else:
        srg_images = images

    images_args = [
        (index, image, image_class, ocr_config)
        for index, (image, image_class) in enumerate(zip(srg_images, image_class_list))
    ]
    preprocessed_images = [None] * len(srg_images)
    # use 5x of CPU count
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        # Submit futures to the executor pool.
        # Map each future back to the arguments used to create that future.
        future_to_args = {
            executor.submit(create_preprocessed_image, image_arg): image_arg
            for image_arg in images_args
        }

        # Images are being preprocessed in worker threads. They will complete in any order.
        for future in concurrent.futures.as_completed(future_to_args):
            image_arg = future_to_args[future]
            try:
                result = future.result()
            # pylint: disable=broad-except
            # we want to log any failure for preprocessing any image.
            except Exception:
                # If the image preprocessing failed we return the original image
                result = image_arg[1]
                logging.warning("Preprocessing image generated an exception")

            preprocessed_images[image_arg[0]] = result

    return preprocessed_images
