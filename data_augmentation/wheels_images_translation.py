import pandas as pd
import cv2

from data_augmentation.utils import get_bound_box, get_random_translation
from utils.preprocessing.constants import ANNOTATIONS_COLUMNS_RETINA_FORMAT,\
    FILE_LOCATION_COLUMN, OBJECT_COLUMN, WHEELS_COLUMN, NUMERIC_COLUMNS


def _augment_object_data_by_translation(train_dataframe, class_column):
    class_data = \
        train_dataframe[train_dataframe[OBJECT_COLUMN] == class_column].copy()

    for row_index, sample_metadata in class_data.iterrows():
        if row_index % 100 == 0:
            print("Augmenting row {}".format(row_index))

        image_array, xmin, xmax, ymin, ymax = \
            get_image_and_bounding_box(sample_metadata)

        translated_image = randomly_translate_image(image_array, xmin,
                                                    xmax, ymin, ymax)
        bounding_box = [xmin, xmax, ymin, ymax]

        translated_image_path = \
            _write_translated_image(sample_metadata, translated_image)

        train_dataframe = \
            _append_translated_metadata(train_dataframe, translated_image_path,
                                        bounding_box, class_column)

    return train_dataframe


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

    train.to_csv('train_retina_format_augmented.csv', header=None, index=False)


if __name__ == '__main__':
    _main()

