import numpy as np

from utils.preprocessing.constants import XMIN_COLUMN, YMIN_COLUMN,\
    XMAX_COLUMN, YMAX_COLUMN


def get_random_translation(image_array, xmin, xmax, ymin, ymax):
    is_horizontal_translation = np.random.randint(0, 2, 1) > 0

    if is_horizontal_translation:
        return get_random_horizontal_translation_matrix(image_array,
                                                        xmin, xmax)
    else:
        return get_random_vertical_translation_matrix(image_array, ymin, ymax)


def get_random_horizontal_translation_matrix(image_array, xmin, xmax):
    rows, columns, _ = image_array.shape
    is_left_translation = np.random.randint(0, 2, 1) > 0
    rand_translation = np.random.randint(0, columns - xmax, 1)
    if is_left_translation:
        rand_translation = -1 * np.random.randint(0, xmin, 1)
    return np.float32([[1, 0, rand_translation],
                       [0, 1, 0]])


def get_random_vertical_translation_matrix(image_array, ymin, ymax):
    rows, columns, _ = image_array.shape
    is_up_translation = np.random.randint(0, 2, 1) > 0
    print("up") if is_up_translation else print("down")
    rand_translation = np.random.randint(0, rows - ymax, 1)
    if is_up_translation:
        rand_translation = -1 * np.random.randint(0, ymin, 1)
    return np.float32([[1, 0, 0],
                       [0, 1, rand_translation]])


def get_bound_box(df_row):
    return int(df_row[XMIN_COLUMN]), int(df_row[XMAX_COLUMN]),\
           int(df_row[YMIN_COLUMN]), int(df_row[YMAX_COLUMN])
