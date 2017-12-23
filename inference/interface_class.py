import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras_retinanet.keras_retinanet.models.resnet import custom_objects
from keras_retinanet.keras_retinanet.preprocessing.csv_generator \
    import CSVGenerator


class ObjectDetectionModel(object):
    def __init__(self, model_checkpoint_path, test_data_path,
                 classes_path, prediction_threshold=0.5):
        self.model_checkpoint_path = model_checkpoint_path
        self.test_data_path = test_data_path
        self.classes_path = classes_path
        self.prediction_threshold = prediction_threshold
        self.model = load_model(model_checkpoint_path,
                                custom_objects=custom_objects)
        self.data_generator = self._build_generator()
        self.data_size = self.data_generator.size()

    def _build_generator(self):
        image_data_generator = keras.preprocessing.image.ImageDataGenerator()
        return CSVGenerator(self.test_data_path, self.classes_path,
                            image_data_generator,
                            batch_size=10)

    def generate_image_and_prediction(self, threshold=None):
        random_image_index = np.random.randint(0, self.data_size, 1)[0]

        print("random index", random_image_index)
        image = self.data_generator.load_image(random_image_index)
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        image = self.data_generator.preprocess_image(image)
        image, scale = self.data_generator.resize_image(image)
        annotations = self.data_generator.load_annotations(random_image_index)
        print("annotations")
        print(annotations)
        _, _, detections = self.model.predict(np.expand_dims(image, axis=0))
        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[
            0, np.arange(detections.shape[1]), 4 + predicted_labels]
        detections[0, :, :4] /= scale

        threshold = self.prediction_threshold if threshold is None else threshold
        for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
            if score < threshold:
                continue
            b = detections[0, idx, :4].astype(int)
            cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
            caption = "{} {:.3f}".format(
                self.data_generator.label_to_name(label), score)
            cv2.putText(draw, caption, (b[0], b[1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
            cv2.putText(draw, caption, (b[0], b[1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

        for annotation in annotations:
            label = int(annotation[4])
            b = annotation[:4].astype(int)
            cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            caption = "{}".format(self.data_generator.label_to_name(label))
            cv2.putText(draw, caption, (b[0], b[1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
            cv2.putText(draw, caption, (b[0], b[1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(draw)
        plt.show()