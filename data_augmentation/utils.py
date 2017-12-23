"""Utilery for data augmentation of images"""
import numpy as np
import cv2

from preprocessing.parse_data.constants import FILE_LOCATION_COLUMN, XMIN_COLUMN,\
    YMIN_COLUMN, XMAX_COLUMN, YMAX_COLUMN


def get_image_and_bounding_box(image_metadata):
    """
    Uses image metadata to read the image and obtain bounding box
    :param image_metadata: panda DataFrame row with corresponding attributes:
            - file_location, xmin, ymin, xmax, ymax
    :return: tuple with image_array and separate bounding box
    """
    image_path = image_metadata[FILE_LOCATION_COLUMN]
    image_array = cv2.imread(image_path)
    xmin, ymin, xmax, ymax = get_bound_box(image_metadata)
    return image_array, xmin, ymin, xmax, ymax


def randomly_translate_image(image_array, xmin, ymin, xmax, ymax):
    """
    Performs either a vertical or horizontal translation of an image
    :param image_array: np.array usually coming the output of an imread method
    :param xmin: int
    :param xmax: int
    :param ymin: int
    :param ymax: int
    :return: np.array representing the translated image
    """
    translation_matrix, xmin, ymin, xmax, ymax = \
        get_random_translation(image_array, xmin, ymin, xmax, ymax)

    rows, columns, _ = image_array.shape
    translated_image = cv2.warpAffine(image_array, translation_matrix,
                                      (columns, rows))

    return translated_image, xmin, ymin, xmax, ymax


def get_random_translation(image_array, xmin, ymin, xmax, ymax):
    """
    Wrapper for random horizontal or vertical translation
    :param image_array: np.array usually coming the output of an imread method
    :param xmin: int
    :param xmax: int
    :param ymin: int
    :param ymax: int
    :return: np.array representing the translation, bounding box per coordinate
    """
    is_horizontal_translation = np.random.randint(0, 2, 1) > 0

    if is_horizontal_translation:
        translation_matrix, xmin, xmax = \
            random_horizontal_translation(image_array, xmin, xmax)

    else:
        translation_matrix, ymin, ymax = \
            random_vertical_translation(image_array, ymin, ymax)

    return translation_matrix, xmin, ymin, xmax, ymax


def random_horizontal_translation(image_array, xmin, xmax):
    """
    Performs horizontal translation on an image, randomly to the left or right,
    redefining the bounding box by the translation
    :param image_array: np.array usually coming the output of an imread method
    :param xmin: int
    :param xmax: int
    :return: np.array of the translation matrix, new xmin and new xmax for
    bounding box
    """
    columns = image_array.shape[1]
    is_left_translation = np.random.randint(0, 2, 1) > 0
    rand_translation = np.random.randint(0, columns - xmax + 1, 1)[0]
    if is_left_translation:
        rand_translation = -1 * np.random.randint(0, xmin, 1)[0]
    translation_matrix = np.float32([[1, 0, rand_translation], [0, 1, 0]])

    xmin += rand_translation
    xmax += rand_translation
    return translation_matrix, int(xmin), int(xmax)


def random_vertical_translation(image_array, ymin, ymax):
    """
    Performs vertical translation on an image, randomly up or down
    redefining the bounding box by the translation
    :param image_array: np.array usually coming the output of an imread method
    :param ymin: int
    :param ymax: int
    :return: np.array of the translation matrix, new ymin and new ymax for
    bounding box
    """
    rows = image_array.shape[0]
    is_up_translation = np.random.randint(0, 2, 1) > 0
    rand_translation = np.random.randint(0, rows - ymax + 1, 1)[0]
    if is_up_translation:
        rand_translation = -1 * np.random.randint(0, ymin, 1)[0]
    translation_matrix = np.float32([[1, 0, 0], [0, 1, rand_translation]])

    ymin += rand_translation
    ymax += rand_translation

    return translation_matrix, int(ymin), int(ymax)


def get_bound_box(df_row):
    """
    Wrapper to quickly get bounding box of per coordinate
    :param df_row: pandas DataFrame row with corresponding attributes:
            - xmin, ymin, xmax, ymax
    :return: tuple of ints defining the bounding box
    """
    return int(df_row[XMIN_COLUMN]), int(df_row[YMIN_COLUMN]),\
           int(df_row[XMAX_COLUMN]), int(df_row[YMAX_COLUMN])
