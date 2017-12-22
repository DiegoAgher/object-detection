import pandas as pd
import cv2

from data_augmentation.utils import get_bound_box, get_random_translation
from utils.preprocessing.constants import ANNOTATIONS_COLUMNS_RETINA_FORMAT,\
    FILE_LOCATION_COLUMN, OBJECT_COLUMN, WHEELS_COLUMN, NUMERIC_COLUMNS

train = pd.read_csv('train_retina_format.csv', header=None,
                    names=ANNOTATIONS_COLUMNS_RETINA_FORMAT)

wheels_data = train[train[OBJECT_COLUMN] == 'wheels'].copy()

for row_index, row in wheels_data.iterrows():
    if row_index % 100 == 0:
        print("Augmenting row {}".format(row_index))
    image_path = row[FILE_LOCATION_COLUMN]
    image_array = cv2.imread(image_path)
    rows, columns, _ = image_array.shape
    xmin, xmax, ymin, ymax = get_bound_box(row)
    translation_matrix, xmin, xmax, ymin, ymax = \
        get_random_translation(image_array, xmin, xmax, ymin, ymax)

    translated_image = cv2.warpAffine(image_array, translation_matrix,
                                      (columns, rows))
    translated_image_path = image_path.replace('.', '_t.')
    cv2.imwrite(translated_image_path, translated_image)
    new_image_row = pd.DataFrame(
        [translated_image_path, xmin, ymin,
         xmax, ymax, WHEELS_COLUMN]).transpose()
    new_image_row.columns = ANNOTATIONS_COLUMNS_RETINA_FORMAT
    train = train.append(new_image_row)


for col in NUMERIC_COLUMNS:
    train[col] = train[col].astype(str)

train.to_csv('train_retina_format_augmented.csv', header=None, index=False)






