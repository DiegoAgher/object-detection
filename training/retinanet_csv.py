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

from training.constants import DESCRIPTION_TRAIN_PARSER


def create_models(num_classes):
    # create "base" model (no NMS)
    image = keras.layers.Input((None, None, 3))
    model = ResNet50RetinaNet(image, num_classes=num_classes,
                              weights='imagenet', nms=False)
    training_model = model

    # append NMS for prediction only
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


def create_callbacks(model, training_model, prediction_model,
                     validation_generator, args):
    callbacks = []

    # save the prediction model
    if args.snapshots:
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                'resnet50_csv_test.h5'
            ),
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


def parse_args(args):
    csv_parser = argparse.ArgumentParser(description=DESCRIPTION_TRAIN_PARSER)

    csv_parser.add_argument('annotations',
                            help='Path to CSV file containing annotations for '
                                 'training.')
    csv_parser.add_argument('classes',
                            help='Path to CSV file containing class label '
                                 'mapping.')
    csv_parser.add_argument('--val-annotations',
                            help='Path to CSV file containing annotations for '
                                 'validation (optional).')

    csv_parser.add_argument('--batch-size',
                            help='Size of the batches.', default=1, type=int)
    csv_parser.add_argument('--epochs',
                            help='Number of epochs to train.',
                            type=int, default=5)
    csv_parser.add_argument('--steps',
                            help='Number of steps per epoch.',
                            type=int, default=1000)
    csv_parser.add_argument('--snapshot-path',
                            help='Path to store snapshots of models during '
                                 'training (defaults to \'./snapshots\')',
                            default='./snapshots')
    csv_parser.add_argument('--no-snapshots',
                            help='Disable saving snapshots.', dest='snapshots',
                            action='store_false')
    csv_parser.set_defaults(snapshots=True)

    return csv_parser.parse_args(args)


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def main(args=None):

    train_generator, validation_generator = create_generators(args)

    print('Creating model, this may take a second...')
    model, training_model, prediction_model = \
        create_models(num_classes=train_generator.num_classes())

    print(model.summary())

    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        args)

    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks)

if __name__ == '__main__':
    main()
