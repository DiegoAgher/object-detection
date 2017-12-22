import numpy as np
import cv2

from utils.preprocessing.constants import FILE_LOCATION_COLUMN, XMIN_COLUMN,\
    YMIN_COLUMN, XMAX_COLUMN, YMAX_COLUMN


def get_image_and_bounding_box(image_metadata):
    image_path = image_metadata[FILE_LOCATION_COLUMN]
    image_array = cv2.imread(image_path)
    xmin, xmax, ymin, ymax = get_bound_box(image_metadata)
    return image_array, xmin, xmax, ymin, ymax


def randomly_translate_image(image_array, xmin, xmax, ymin, ymax):
    translation_matrix, xmin, xmax, ymin, ymax = \
        get_random_translation(image_array, xmin, xmax, ymin, ymax)
    rows, columns, _ = image_array.shape

    return cv2.warpAffine(image_array, translation_matrix, (columns, rows))


def get_random_translation(image_array, xmin, xmax, ymin, ymax):
    is_horizontal_translation = np.random.randint(0, 2, 1) > 0

    if is_horizontal_translation:
        translation_matrix, xmin, xmax = \
            random_horizontal_translation(image_array, xmin, xmax)

    else:
        translation_matrix, ymin, ymax = \
            random_vertical_translation(image_array, ymin, ymax)

    return translation_matrix, xmin, xmax, ymin, ymax


def random_horizontal_translation(image_array, xmin, xmax):
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
    return int(df_row[XMIN_COLUMN]), int(df_row[XMAX_COLUMN]),\
           int(df_row[YMIN_COLUMN]), int(df_row[YMAX_COLUMN])
