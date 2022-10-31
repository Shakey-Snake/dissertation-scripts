# importing the necessary packages
import gc
import os
import sys
import tempfile
from csv import writer
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_model_optimization as tfmot
from keras import regularizers
from keras.layers import Input, Dense, GlobalAveragePooling2D, Activation, Dropout
from keras.metrics import SparseTopKCategoricalAccuracy, TopKCategoricalAccuracy
from keras_preprocessing.image import ImageDataGenerator
from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_low_magnitude

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if sys.argv[1] == 'ResNet152V2':
    pretrained_model = tf.keras.applications.ResNet152V2
    model_image_size = 224
    preprocess_function = tf.keras.applications.resnet_v2.preprocess_input
    keras_model = "ResNet152V2"
elif sys.argv[1] == 'MobileNetV3Large':
    pretrained_model = tf.keras.applications.MobileNetV3Large
    model_image_size = 224
    preprocess_function = tf.keras.applications.mobilenet_v3.preprocess_input
    keras_model = "MobileNetV3Large"
elif sys.argv[1] == 'InceptionV3':
    pretrained_model = tf.keras.applications.InceptionV3
    model_image_size = 299
    preprocess_function = tf.keras.applications.inception_v3.preprocess_input
    keras_model = "InceptionV3"
elif sys.argv[1] == 'DenseNet201':
    pretrained_model = tf.keras.applications.DenseNet201
    model_image_size = 224
    preprocess_function = tf.keras.applications.densenet.preprocess_input
    keras_model = "DenseNet201"
else:
    raise ValueError('Supply a model of the following: ResNet152V2, MobileNetV3Large, InceptionV3, DenseNet201')

# check if we need base model 0 for false 1 for true

typesList = []

# if sys.argv[2] == '1':
#     typesList.append('base_model')

for x in range(0, 100, 10):
    for y in range(x + 10, 100, 10):
        typesList.append(['prune', x / 100, y / 100])

for x in range(0, 100, 10):
    typesList.append(['prune', x, 0.95])
    typesList.append(['prune', x, 0.99])
    typesList.append(['prune', x, 0.999])
    typesList.append(['prune', x, 0.9999])


print(typesList)
outType = 'both'

# check if we need models, stats or both (0, 1, 2)
if sys.argv[3] == '0':
    outType = 'models'
elif sys.argv[3] == '1':
    outType = 'stats'

if keras_model != 'MobileNetV3Large':
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        fill_mode='nearest',
        dtype='float32',
        # preprocessing_function=preprocess_function,
        # rotation_range=40,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)
else:
    train_datagen = ImageDataGenerator(
        # rescale = 1. / 255,
        fill_mode='nearest',
        dtype='float32',
        # preprocessing_function=preprocess_function,
        # rotation_range=40,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(  # rescale=1. / 255
    )

train = train_datagen.flow_from_directory(
    './food101/train',  # this is the target directory
    target_size=(model_image_size, model_image_size),
    batch_size=32,
    class_mode='sparse')

val = test_datagen.flow_from_directory(
    './food101/test',  # this is the target directory
    target_size=(model_image_size, model_image_size),
    batch_size=32,
    class_mode='sparse')

for pruningType in typesList:
    for i in range(3):
        if pruningType[0] == 'prune':
            model_name = keras_model + '_' + pruningType[0] + '_' + str(pruningType[1]) + '_' + str(pruningType[2]) \
                         + '_' + str(i)
        else:
            model_name = keras_model + '_' + pruningType + '_' + str(i)

        base_model = pretrained_model(include_top=False, weights='imagenet',
                                      input_shape=(model_image_size, model_image_size, 3))
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D(name='poolingLayer')(x)
        x = Dropout(0.2)(x)
        x = Dense(101, name='outputLayer')(x)
        outputs = Activation(activation="softmax",
                             dtype=tf.float32,
                             name='activationLayer')(x)

        model = tf.keras.Model(inputs=base_model.input, outputs=outputs, name=model_name)

        # model.summary()

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.SGD(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), SparseTopKCategoricalAccuracy()])

        fileToSendTo = './logs/' + keras_model

        hist_model = model.fit(train,
                               epochs=5,
                               validation_data=val)

        model_results = model.evaluate(val)

        logdir = tempfile.mkdtemp()

        end_step = np.ceil(75750 / 16).astype(np.int32) * 2

        if pruningType[0] == 'prune':
            # Define model for pruning.
            print('pruning')
            if keras_model == 'MobileNetV3Large':
                pruning_params = {
                    # 'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=pruningType[1],
                    #                                                           begin_step=0,
                    #                                                           end_step=-1)
                    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=pruningType[1],
                                                                             final_sparsity=pruningType[2],
                                                                             begin_step=0,
                                                                             end_step=end_step)
                }


                def apply_pruning_to_conv(layer):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
                    return layer


                # Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense`
                # to the layers of the model.
                model = tf.keras.models.clone_model(
                    model,
                    clone_function=apply_pruning_to_conv,
                )

            else:
                pruning_params = {
                    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=pruningType[1],
                                                                             final_sparsity=pruningType[2],
                                                                             begin_step=0,
                                                                             end_step=end_step)
                }

                model = prune_low_magnitude(model, **pruning_params)

            callbacks = [
                tfmot.sparsity.keras.UpdatePruningStep(),
                tfmot.sparsity.keras.PruningSummaries(log_dir=logdir)
            ]
        else:
            callbacks = []

        base_model.trainable = True

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.SGD(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), SparseTopKCategoricalAccuracy()])

        hist_model = model.fit(train,
                               epochs=2,
                               validation_data=val,
                               callbacks=[callbacks])

        result = model.evaluate(val)


        def get_gzipped_model_size(file):
            # Returns size of gzipped model, in bytes.
            import zipfile

            _, zipped_file = tempfile.mkstemp('.zip')
            with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
                f.write(file)

            size = os.path.getsize(zipped_file)

            os.close(_)
            return size


        if pruningType[0] == 'prune':
            model = tfmot.sparsity.keras.strip_pruning(model)

        # convert to tf lite model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        tflite_file = Path('./models/' + model_name + '.tflite')
        tflite_file.write_bytes(tflite_model)

        model_size = get_gzipped_model_size(tflite_file)
        del tflite_file
        gc.collect()

        # for image_batch in train[0]:
        #     print(image_batch)


        def representative_dataset():
            if keras_model != 'MobileNetV3Large':
                test_datagen = ImageDataGenerator(rescale=1. / 255)
            else:
                test_datagen = ImageDataGenerator(  # rescale=1. / 255
                )

            test_generator = test_datagen.flow_from_directory(
                './food101/test',  # this is the target directory
                target_size=(model_image_size, model_image_size),
                batch_size=1,
                class_mode='sparse')

            for ind in range(100):
                img_with_label = test_generator.next()
                yield [np.array(img_with_label[0], dtype=np.float32, ndmin=2)]


        quant_converter = tf.lite.TFLiteConverter.from_keras_model(model)
        quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quant_converter.representative_dataset = representative_dataset
        # Ensure that if any ops can't be quantized, the converter throws an error
        quant_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Set the input and output tensors to uint8 (APIs added in r2.3)
        quant_converter.inference_input_type = tf.int8
        quant_converter.inference_output_type = tf.uint8

        print('converting to quantized tflite model')
        tflite_quant_model = quant_converter.convert()

        print('saving quantized tflite model')
        quantized_tflite_file = Path('./models/quant_' + model_name + '.tflite')
        quantized_tflite_file.write_bytes(tflite_quant_model)

        interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
        input_type = interpreter.get_input_details()[0]['dtype']
        print('input: ', input_type)
        output_type = interpreter.get_output_details()[0]['dtype']
        print('output: ', output_type)

        quant_model_size = get_gzipped_model_size(quantized_tflite_file)
        del quantized_tflite_file
        gc.collect()

        # output results to csv file
        if sys.argv[3] == '0' or sys.argv[3] == '2':
            row_output = [model_name, str(result[0]), str(result[1]),
                          str(result[2]), str(model_size), str(quant_model_size)]
            with open(f'{keras_model}.csv', 'a') as fd:
                writer_object = writer(fd)
                writer_object.writerow(row_output)
