"""Script used to augment data for wheels class for training RetinaNet """
import pandas as pd
import cv2

from data_augmentation.utils import get_image_and_bounding_box,\
    randomly_translate_image
from preprocessing.parse_data.constants import ANNOTATIONS_COLUMNS_RETINA_FORMAT,\
    FILE_LOCATION_COLUMN, OBJECT_COLUMN, WHEELS_COLUMN, NUMERIC_COLUMNS


def _augment_object_data_by_translation(train_dataframe, class_column):
    class_data = \
        train_dataframe[train_dataframe[OBJECT_COLUMN] == class_column].copy()
    for row_index, sample_metadata in class_data.iterrows():
        if row_index % 100 == 0:
            print("Augmenting row {}".format(row_index))

        image_array, xmin, ymin, xmax, ymax = \
            get_image_and_bounding_box(sample_metadata)

        translated_image, xmin_t, ymin_t, xmax_t, ymax_t =\
            randomly_translate_image(image_array, xmin, ymin, xmax, ymax)
        translated_bounding_box = [xmin_t, ymin_t, xmax_t, ymax_t]

        translated_image_path = \
            _write_translated_image(sample_metadata, translated_image)

        train_dataframe = \
            _append_translated_metadata(train_dataframe, translated_image_path,
                                        translated_bounding_box, class_column)

    return train_dataframe


def _write_translated_image(image_metadata, translated_image):
    translated_image_path = \
        image_metadata[FILE_LOCATION_COLUMN].replace('.', '_t.')
    cv2.imwrite(translated_image_path, translated_image)
    return translated_image_path


def _append_translated_metadata(dataframe, path_location, bounding_box,
                                class_column):
    xmin, ymin, xmax, ymax = bounding_box
    new_image_row = pd.DataFrame(
        [path_location, xmin, ymin,
         xmax, ymax, class_column]).transpose()
    new_image_row.columns = ANNOTATIONS_COLUMNS_RETINA_FORMAT
    return dataframe.append(new_image_row)


def _main():
    train = pd.read_csv('train_retina_format.csv', header=None,
                        names=ANNOTATIONS_COLUMNS_RETINA_FORMAT)

    train = _augment_object_data_by_translation(train, WHEELS_COLUMN)

    train.fillna('', inplace=True)
    for col in NUMERIC_COLUMNS:
        train[col] = train[col].astype(str)
        train[col] = train[col].apply(lambda x: x.replace(".0", ''))

    train.to_csv('train_retina_format_augmented.csv', header=None, index=False)


if __name__ == '__main__':
    _main()
