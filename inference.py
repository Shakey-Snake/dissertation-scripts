from _csv import writer

import numpy as np
from tensorflow import keras
import tensorflow as tf
import os
import time
from PIL import Image

# get a list of the models
# folder path
dir_path = './models/'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Iterate directory
models = os.listdir(dir_path)
print(models)
models.sort()
print(models)

for model_file in models:
    # for each model
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=dir_path + model_file)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    print(input_details)
    # for 100 times
    inf_list = []
    for i in range(100):
        # get a picture from the dataset test folder
        data_dir_path = '../data/food101/test/french_fries/1008163.jpg'

        # print(model_file)

        model_name = model_file.split("_")

        if model_name[0] == 'quant':
            if model_name[1] == 'DenseNet201' or model_name[1] == 'MobileNetV3Large' or model_name[0] == 'ResNet152V2':
                img_height, img_width = 224, 224
            elif model_name[1] == 'InceptionV3':
                img_height, img_width = 299, 299

            img = tf.keras.utils.load_img(data_dir_path, target_size=[img_height, img_width])

            # print('quant')
            test_image = np.expand_dims(img, axis=0).astype(np.uint8)

        else:
            if model_name[0] == 'DenseNet201' or model_name[0] == 'MobileNetV3Large' or model_name[0] == 'ResNet152V2':
                img_height, img_width = 224, 224
            elif model_name[0] == 'InceptionV3':
                img_height, img_width = 299, 299
            img = tf.keras.utils.load_img(data_dir_path, target_size=[img_height, img_width])

            test_image = np.expand_dims(img, axis=0).astype(np.float32)

        # Test the model on random input data.
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        interpreter.set_tensor(input_index, test_image)

        # start a timer
        start = time.perf_counter()
        # evaluate
        interpreter.invoke()
        # end the timer
        end = time.perf_counter()

        inference_time = end - start

        # print(inference_time)

        inf_list.append(inference_time)

    average = sum(inf_list) / len(inf_list)
    print(average)

    # add to a csv file
    row_output = [model_file, 'average_inference_timings: ' + str(average)]
    with open('inference_timings.csv', 'a') as fd:
        writer_object = writer(fd)
        writer_object.writerow(row_output)
