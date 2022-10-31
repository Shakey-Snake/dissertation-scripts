# create a bash file that will perform inference on
# all models in the model file

# Iterate directory
import os

#dir_path = './models/'

dir_path = './models/'

models = os.listdir(dir_path)
print(models)
models.sort()
print(models)

f = open("./tensorflow-master/inferencescript.sh", "w")

for model in models:

    model_name = model.split(".")
    if len(model_name) > 2:
        model_name = model_name[0] + model_name[1] + model_name[2]
    else:
        model_name = model_name[0]
    f.write(f'bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model --graph=../models/{model} --num_threads=-1 '
            f'--use_xnnpack=false --enable_op_profiling=true --report_peak_memory_footprint=true'
            f' --allow_dynamic_profiling_buffer_increase=true' 
            f' --profiling_output_csv_file=../models/data/inference_{model_name}.csv \n')
f.close()
