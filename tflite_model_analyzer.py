import csv

import numpy as np
import pandas as pd
import tensorflow as tf

# tf.lite.experimental.Analyzer.analyze(model_path='./models/quant_DenseNet201_base_model_0.tflite',
#                                       model_content=None,
#                                       gpu_compatibility=False)

# Iterate directory
import os

from keras_preprocessing.image import ImageDataGenerator

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

dir_path = './models/quant/'

models = os.listdir(dir_path)
models.sort()

batch_size = 64

# for model in models:
#     model_name = model.split(".")
#     model_args = model_name[0].split("_")

for model in models:
    model_name = model.split(".")
    model_args = model_name[0].split("_")

    model_name = model_name[0]+model_name[1]+model_name[2]

    if model_args[0] == 'ResNet152V2' or model_args[1] == 'ResNet152V2':
        pretrained_model = tf.keras.applications.ResNet152V2
        model_image_size = 224
        preprocess_function = tf.keras.applications.resnet_v2.preprocess_input
        keras_model = "ResNet152V2"
    elif model_args[0] == 'MobileNetV3Large' or model_args[1] == 'MobileNetV3Large':
        pretrained_model = tf.keras.applications.MobileNetV3Large
        model_image_size = 224
        preprocess_function = tf.keras.applications.mobilenet_v3.preprocess_input
        keras_model = "MobileNetV3Large"
    elif model_args[0] == 'InceptionV3' or model_args[1] == 'InceptionV3':
        pretrained_model = tf.keras.applications.InceptionV3
        model_image_size = 299
        preprocess_function = tf.keras.applications.inception_v3.preprocess_input
        keras_model = "InceptionV3"
    elif model_args[0] == 'DenseNet201' or model_args[1] == 'DenseNet201':
        pretrained_model = tf.keras.applications.DenseNet201
        model_image_size = 224
        preprocess_function = tf.keras.applications.densenet.preprocess_input
        keras_model = "DenseNet201"

    print(keras_model)

    if model_args[0] == 'quant':
        if model_args[1] == 'DenseNet201' or model_args[1] == 'InceptionV3' or model_args[1] == 'MobileNetV3Large' \
                or model_args[1] == 'ResNet152V2':
            test_datagen = ImageDataGenerator(rescale=1. / 255, dtype='int8')
        else:
            test_datagen = ImageDataGenerator(rescale=1. / 255, dtype='uint8')
    else:
        test_datagen = ImageDataGenerator(rescale=1. / 255)

    val = test_datagen.flow_from_directory(
        '../data/food101/test',  # this is the target directory
        target_size=(model_image_size, model_image_size),
        batch_size=batch_size,
        class_mode='sparse')

    dataset_labels = sorted(val.class_indices.items(), key=lambda pair: pair[1])
    dataset_labels = np.array([key.title() for key, value in dataset_labels])
    # print(dataset_labels)

    # Load TFLite model and see some details about input/output

    tflite_interpreter = tf.lite.Interpreter(model_path=f'{dir_path}{model}')

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    print("== Input details ==")
    print("name:", input_details[0]['name'])
    print("shape:", input_details[0]['shape'])
    print("type:", input_details[0]['dtype'])

    print("\n== Output details ==")
    print("name:", output_details[0]['name'])
    print("shape:", output_details[0]['shape'])
    print("type:", output_details[0]['dtype'])

    tflite_interpreter.resize_tensor_input(input_details[0]['index'],
                                           (batch_size, model_image_size, model_image_size, 3))
    tflite_interpreter.resize_tensor_input(output_details[0]['index'], (batch_size, 101))
    tflite_interpreter.allocate_tensors()

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    # print("== Input details ==")
    # print("name:", input_details[0]['name'])
    # print("shape:", input_details[0]['shape'])
    # print("type:", input_details[0]['dtype'])
    #
    # print("\n== Output details ==")
    # print("name:", output_details[0]['name'])
    # print("shape:", output_details[0]['shape'])
    # print("type:", output_details[0]['dtype'])

    correct = 0
    incorrect = 0
    top_5_correct = 0
    top_5_incorrect = 0

    for i in range(16):
        val_image_batch, val_label_batch = val.next()

        # print(val_label_batch)
        true_label_ids = val_label_batch

        tflite_interpreter.set_tensor(input_details[0]['index'], val_image_batch)

        tflite_interpreter.invoke()

        tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
        # print("Prediction results shape:", tflite_model_predictions.shape)

        # top-1 accuracy
        tflite_predicted_ids = np.argmax(tflite_model_predictions, axis=-1)
        # print(tflite_model_predictions[0])
        # print(tflite_predicted_ids)
        tflite_predicted_labels = dataset_labels[tflite_predicted_ids]
        tflite_label_id = np.argmax(val_label_batch, axis=-1)

        # top-5

        tflite_predicted_ids_5 = np.argpartition(tflite_model_predictions, -5, axis=-1)[:, -5:]
        # print(tflite_predicted_ids)
        #
        # print(tflite_predicted_ids_5[0])
        # print(tflite_predicted_ids_5)

        # print(tflite_predicted_ids_5[1])
        # print(tflite_predicted_ids_5[2])
        # print(tflite_predicted_ids_5[3])
        # print(tflite_predicted_ids_5[4])

        # print(true_label_ids)

        # try a large number of
        for n in range(batch_size):
            top_5 = False
            for idx, tf_iter in enumerate(tflite_predicted_ids_5[n]):
                # print(tf_iter)
                if tf_iter == true_label_ids[n]:
                    top_5_correct += 1
                    top_5 = True
                    break
            if not top_5:
                # print('incorrect')
                top_5_incorrect += 1
            if tflite_predicted_ids[n] == true_label_ids[n]:
                correct += 1
            else:
                incorrect += 1

    accuracy = correct / (correct + incorrect)
    top_5_accuracy = top_5_correct / (top_5_correct + top_5_incorrect)
    with open("tflite_accuracy.csv", 'a') as f:
        data_row = [model, accuracy, top_5_accuracy]
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerow(data_row)

    print(f"accuracy:{accuracy}")
    print(f"accuracy top-5:{top_5_accuracy}")
