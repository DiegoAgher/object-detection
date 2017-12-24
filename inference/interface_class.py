"""Implementation of a class as interface to perform inference of object
detection"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras.preprocessing as K_preprocessing
from keras.models import load_model
from keras_retinanet.keras_retinanet.models.resnet import custom_objects
from keras_retinanet.keras_retinanet.preprocessing.csv_generator \
    import CSVGenerator


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
        image_data_generator = K_preprocessing.image.ImageDataGenerator()
        return CSVGenerator(self.test_data_path, self.classes_path,
                            image_data_generator,
                            batch_size=10)

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
        self._index += 1
        return image, draw, annotations

    def _rescale_image(self, image):
        image = self.data_generator.preprocess_image(image)
        return self.data_generator.resize_image(image)

    def _get_predictions(self, image, scale):
        _, _, detections = self.model.predict(np.expand_dims(image, axis=0))
        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(detections.shape[1]),
                            4 + predicted_labels]
        detections[0, :, :4] /= scale
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
        current_detection = detections[0, prediction_id, :4].astype(int)
        cv2.rectangle(draw, (current_detection[0], current_detection[1]),
                      (current_detection[2], current_detection[3]),
                      (0, 0, 255), 3)

        caption = "{} {:.3f}".format(
            self.data_generator.label_to_name(label), score)
        cv2.putText(draw, caption,
                    (current_detection[0], current_detection[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
        cv2.putText(draw, caption,
                    (current_detection[0], current_detection[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

    def _draw_annotations(self, draw, annotation):
        label = int(annotation[4])
        annotation_coords = annotation[:4].astype(int)
        cv2.rectangle(draw, (annotation_coords[0], annotation_coords[1]),
                      (annotation_coords[2], annotation_coords[3]),
                      (0, 255, 0), 2)

        caption = "{}".format(self.data_generator.label_to_name(label))
        cv2.putText(draw, caption,
                    (annotation_coords[0], annotation_coords[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
        cv2.putText(draw, caption,
                    (annotation_coords[0], annotation_coords[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
