"""Implementation of a class as interface to perform inference of object
detection"""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.preprocessing.image as image_preprocessing
from keras.models import load_model
from keras_retinanet.keras_retinanet.models.resnet import custom_objects
from keras_retinanet.keras_retinanet.preprocessing.csv_generator \
    import CSVGenerator
from inference.constants import NUMBER_OF_POINTS_IN_BOUNDING_BOX, XMIN_COORD, \
    XMAX_COORD, YMIN_COORD, YMAX_COORD, SCORE_THRESHOLD
from inference.utils import draw_caption
from preprocessing.parse_data.constants import \
    ANNOTATIONS_COLUMNS_RETINA_FORMAT


class ObjectDetectionModel(object):
    """ Class to make inference over images given a generator and a saved model.
    Predict bounding boxes of classes use method
    generate_image_and_prediction"""
    def __init__(self, model_checkpoint_path, test_data_path,
                 classes_path):
        self.model_checkpoint_path = model_checkpoint_path
        self.test_data_path = test_data_path
        self.classes_path = classes_path
        self.model = load_model(model_checkpoint_path,
                                custom_objects=custom_objects)
        self.data_generator = self._build_generator()
        self.data_size = self.data_generator.size()
        self._index = 0

    def _build_generator(self):
        data_path = self.test_data_path
        image_data_generator = image_preprocessing.ImageDataGenerator()
        return CSVGenerator(data_path, self.classes_path,
                            image_data_generator,
                            batch_size=32)

    def generate_image_and_prediction(self, threshold=0.5):
        """
        Plots image along with annotations and predictions: bounding boxes and
        classes depending on probability threshold
        :param threshold: int
        """
        image, draw, annotations = self.get_next_sample()
        image, scale = self._rescale_image(image)

        scores, detections, predicted_labels = \
            self._get_predictions(image, scale)

        self._draw_annotations_and_predictions(draw, annotations,
                                               predicted_labels, scores,
                                               threshold, detections)

    def get_next_sample(self):
        """
        Generates data of next sample: image, draw object, annotations
        :return: tuple of np.arrays
        """
        image = self.data_generator.load_image(self._index)
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        annotations = self.data_generator.load_annotations(self._index)
        self._update_index()
        return image, draw, annotations

    def _update_index(self):
        self._index += 1
        if self._index > self.data_size - 1:
            self._index = 0

    def _rescale_image(self, image):
        image = self.data_generator.preprocess_image(image)
        return self.data_generator.resize_image(image)

    def _get_predictions(self, image, scale):
        _, _, detections = self.model.predict(np.expand_dims(image, axis=0))

        predicted_labels = np.argmax(
            detections[0, :, NUMBER_OF_POINTS_IN_BOUNDING_BOX:], axis=1)

        scores = detections[0,
                            np.arange(detections.shape[1]),
                            NUMBER_OF_POINTS_IN_BOUNDING_BOX + predicted_labels]

        detections[0, :, :NUMBER_OF_POINTS_IN_BOUNDING_BOX] /= scale
        return scores, detections, predicted_labels

    def _draw_annotations_and_predictions(self, draw, annotations,
                                          predicted_labels, scores,
                                          threshold, detections):

        for prediction_id, (label, score) in enumerate(zip(predicted_labels,
                                                           scores)):
            if score < threshold:
                continue
            self._draw_predicted_labels_and_scores(draw, detections,
                                                   prediction_id, label, score)

        for annotation in annotations:
            self._draw_annotations(draw, annotation)

        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(draw)
        plt.show()

    def _draw_predicted_labels_and_scores(self, draw, detections,
                                          prediction_id, label, score):

        bounding_box = (detections[0, prediction_id,
                                   :NUMBER_OF_POINTS_IN_BOUNDING_BOX].
                        astype(int))

        cv2.rectangle(draw,
                      (bounding_box[XMIN_COORD], bounding_box[YMIN_COORD]),
                      (bounding_box[XMAX_COORD], bounding_box[YMAX_COORD]),
                      (0, 0, 255), 3)

        caption = "{} {:.3f}".format(self.data_generator.label_to_name(label),
                                     score)
        draw_caption(draw, caption, bounding_box)

    def _draw_annotations(self, draw, annotation):
        label = int(annotation[4])
        annotation_coords = (annotation[:NUMBER_OF_POINTS_IN_BOUNDING_BOX].
                             astype(int))

        cv2.rectangle(draw,
                      (annotation_coords[XMIN_COORD],
                       annotation_coords[YMIN_COORD]),
                      (annotation_coords[XMAX_COORD],
                       annotation_coords[YMAX_COORD]),
                      (0, 255, 0), 2)

        caption = "{}".format(self.data_generator.label_to_name(label))

        draw_caption(draw, caption, annotation_coords)

    def predict_dataset(self):
        """
        Predicts all samples in dataset provided in self.test_data_path and
        writes a CSV file with predictions.
        :return:
        """
        dataset_prediction = []
        for index, image_location in enumerate(
                self.data_generator.image_data.keys()):

            if index % 200 == 0:
                print("predicting image {}".format(index))

            image = self.data_generator.load_image(index)
            image, scale = self._rescale_image(image)
            scores, detections, predicted_labels = self._get_predictions(
                image, scale)

            for prediction_id, (label, score) in enumerate(
                    zip(predicted_labels, scores)):

                sample_row = [image_location]
                if score >= SCORE_THRESHOLD:
                    sample_row =\
                        self._build_prediction_row(sample_row, prediction_id,
                                                   label, detections)

                dataset_prediction.append(sample_row)

        dataset_prediction = pd.DataFrame(
            dataset_prediction, columns=ANNOTATIONS_COLUMNS_RETINA_FORMAT)

        dataset_prediction.to_csv(
            'inference/predictions.csv'.format(self.model_checkpoint_path),
            header=False, index=False)

    def _build_prediction_row(self, row, prediction_id, label,
                              detections):

        bounding_box =\
            (detections[0, prediction_id, :NUMBER_OF_POINTS_IN_BOUNDING_BOX].
             astype(int))
        label_str = self.data_generator.label_to_name(label)
        row.extend(bounding_box)
        row.append(label_str)
        return row
