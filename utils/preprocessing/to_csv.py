"""Module used to write train, validation and test csv files in RETINANET
friendly format from raw data"""

import os
import xml.etree.ElementTree as elemTree
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.preprocessing.constants import ANNOTATIONS_COLUMNS_RETINA_FORMAT,\
    DATA_DIRECTORY, JPG_SUFFIX, PARSED_DATAFRAME_COLUMNS, XML_SUFFIX
from utils.preprocessing.constants import XMIN_COLUMN, YMIN_COLUMN,\
    XMAX_COLUMN, YMAX_COLUMN, OBJECT_COLUMN


def create_metadata_csv_retinanet():
    """
    Writes train, validation and test csv files with RETINANET format
    """
    data = []
    for root, _, files in os.walk(DATA_DIRECTORY):
        valid_files = [file for file in files
                       if not file.endswith('.DS_Store')]

        for file in valid_files:
            if file.endswith(JPG_SUFFIX):

                file_location = os.path.join(root, file)
                base_row = [file_location]

                xml_file = file.replace(JPG_SUFFIX, XML_SUFFIX)

                if xml_file in files:
                    objects_in_xml = _get_objects_in_xml(root, xml_file)
                    for object_node in objects_in_xml:
                        data.append(base_row + _parse_objects(object_node))
                else:
                    data.append(base_row)

    data_df = pd.DataFrame(data, columns=PARSED_DATAFRAME_COLUMNS)
    data_df.fillna('', inplace=True)

    _write_subset_csvs(data_df)


def _parse_objects(object_node):
    object_name = object_node.find('name').text
    bound_box = object_node.find('bndbox')
    xmin = bound_box.find(XMIN_COLUMN).text
    xmax = bound_box.find(XMAX_COLUMN).text
    ymin = bound_box.find(YMIN_COLUMN).text
    ymax = bound_box.find(YMAX_COLUMN).text

    return [xmin, ymin, xmax, ymax, object_name]


def _get_objects_in_xml(root_dir, xml_file_path):
    xml_file_location = os.path.join(root_dir, xml_file_path)
    parsed_file = elemTree.parse(xml_file_location)
    return parsed_file.findall(OBJECT_COLUMN)


def _write_subset_csvs(dataset):
    train_val_test_sets = train_val_test_split(dataset)

    for set_name, subset in train_val_test_sets.items():
        for_csv_df = subset[ANNOTATIONS_COLUMNS_RETINA_FORMAT]
        csv_name = '{}_retina_format.csv'.format(set_name)
        print("Writing csv file {}".format(csv_name))
        for_csv_df.to_csv(csv_name, header=False, index=False)


def train_val_test_split(arrays, val_test_size=0.15):
    """ Splits dataframe into train, validation and test subsets

    :param arrays: from sklearn doc:
                Allowed inputs are lists, numpy arrays, scipy-sparse matrices
                or pandas dataframes.
    :param val_test_size: size for validation and test subsets
    :return: dictionary with subsets associated to the corresponding key
    """
    sets_dictionary = dict()
    val_test_size *= 2

    train, val_test = train_test_split(arrays, test_size=val_test_size)
    sets_dictionary['train'] = train

    val, test = train_test_split(val_test, test_size=0.5)
    sets_dictionary['val'] = val
    sets_dictionary['test'] = test

    return sets_dictionary

if __name__ == '__main__':
    create_metadata_csv_retinanet()
