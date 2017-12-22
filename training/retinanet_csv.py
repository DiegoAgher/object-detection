"""Script used to train RetinaNet model providing params from command line"""
import sys
import argparse
import os
import keras
import keras.preprocessing.image as image_preprocessing
import tensorflow as tf

from keras_retinanet.keras_retinanet import losses, layers

from keras_retinanet.\
    keras_retinanet.callbacks import RedirectModel
from keras_retinanet.\
    keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.\
    keras_retinanet.models.resnet import ResNet50RetinaNet

from training.constants import DESCRIPTION_TRAIN_PARSER, MODEL_CHECKPOINT_PATH


def create_models(num_classes):
    """
    Builds compiled Retinanet using ReNet50 for object detection
    :param num_classes: number of classes to possibly detect
    :return: base model (model), training and inference model
    """

    image = keras.layers.Input((None, None, 3))
    model = ResNet50RetinaNet(image, num_classes=num_classes,
                              weights='imagenet', nms=False)
    training_model = model

    # include Non Max Supression layer for prediction model only
    classification = model.outputs[1]
    detections = model.outputs[2]
    boxes = keras.layers.Lambda(lambda x: x[:, :, :4])(detections)
    detections = layers.NonMaximumSuppression(name='nms')([boxes,
                                                           classification,
                                                           detections])
    prediction_model = keras.models.Model(inputs=model.inputs,
                                          outputs=model.outputs[:2] +
                                          [detections])

    training_model.compile(
        loss={
            'regression': losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

    return model, training_model, prediction_model


def _create_callbacks(prediction_model, args):
    callbacks = []

    # save the prediction model
    if args.snapshots:
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                MODEL_CHECKPOINT_PATH.format(args.steps, args.epochs)
            ),
            save_best_only=True,
            verbose=1
        )
        checkpoint = RedirectModel(checkpoint, prediction_model)
        callbacks.append(checkpoint)

    lr_scheduler = (keras.callbacks.
                    ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2,
                                      verbose=1, mode='auto', epsilon=0.0001,
                                      cooldown=0, min_lr=0))
    callbacks.append(lr_scheduler)

    return callbacks


def create_generators(args):
    """
    Instances train and validation CSV ImageData Generators, data augmentation
    for vertical and horizontal flipping is set automatically.
    :param args: parsed arguments from command line
    :return: train and validations generators
    """
    train_image_data_generator = (image_preprocessing.
                                  ImageDataGenerator(vertical_flip=True,
                                                     horizontal_flip=True))
    val_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

    train_generator = CSVGenerator(
        args.annotations,
        args.classes,
        train_image_data_generator,
        batch_size=args.batch_size
    )

    if args.val_annotations:
        validation_generator = CSVGenerator(
            args.val_annotations,
            args.classes,
            val_image_data_generator,
            batch_size=args.batch_size
        )
    else:
        validation_generator = None

    return train_generator, validation_generator


def _parse_args(args):

    csv_parser = argparse.ArgumentParser(description=DESCRIPTION_TRAIN_PARSER)

    csv_parser.add_argument('--annotations',
                            help='Path to CSV file containing annotations for '
                                 'training.')
    csv_parser.add_argument('--classes',
                            help='Path to CSV file containing class label '
                                 'mapping.')
    csv_parser.add_argument('--val-annotations',
                            help='Path to CSV file containing annotations for '
                                 'validation (optional).')

    csv_parser.add_argument('--batch-size',
                            help='Size of the batches.', default=10, type=int)
    csv_parser.add_argument('--epochs',
                            help='Number of epochs to train.',
                            type=int, default=5)
    csv_parser.add_argument('--steps',
                            help='Number of steps per epoch.',
                            type=int, default=1000)
    csv_parser.add_argument('--snapshot-path',
                            help='Path to store snapshots of models during '
                                 'training (defaults to \'./snapshots\')',
                            default='keras_retinanet/snapshots')
    csv_parser.add_argument('--no-snapshots',
                            help='Disable saving snapshots.', dest='snapshots',
                            action='store_false')
    csv_parser.set_defaults(snapshots=True)

    return csv_parser.parse_args(args)


def _get_session():
    config = tf.ConfigProto()
    return tf.Session(config=config)


def _main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)

    keras.backend.tensorflow_backend.set_session(_get_session())
    train_generator, validation_generator = create_generators(args)

    print('Creating model, this may take a second...')
    model, training_model, prediction_model = \
        create_models(num_classes=train_generator.num_classes())

    print(model.summary())

    callbacks = _create_callbacks(prediction_model, args)

    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=50)

if __name__ == '__main__':
    _main()
