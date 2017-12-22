import numpy as np
import pandas as pd
import cv2

from data_augmentation.utils import get_bound_box, get_random_translation
from utils.preprocessing.constants import ANNOTATIONS_COLUMNS_RETINA_FORMAT,\
    FILE_LOCATION_COLUMN, OBJECT_COLUMN

from utils.preprocessing.constants import XMIN_COLUMN, YMIN_COLUMN,\
    XMAX_COLUMN, YMAX_COLUMN


train = pd.read_csv('train_retina_format.csv', header=None,
                    names=ANNOTATIONS_COLUMNS_RETINA_FORMAT)
train.fillna('', inplace=True)

wheels_data = train[train[OBJECT_COLUMN] == 'wheels']

for _, row in wheels_data.iterrows():
    image_path = row[FILE_LOCATION_COLUMN]
    image_array = cv2.imread(image_path)
    rows, cols, _ = image_array.shape
    xmin, xmax, ymin, ymax = get_bound_box(row)
    M_t = get_random_translation(image_array, xmin, xmax, ymin, ymax)
    dst = cv2.warpAffine(image_array, M_t, (cols, rows))




