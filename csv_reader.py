import csv
import os

dir_path = './models/data'

models = os.listdir(dir_path)
print(models)
models.sort()
print(models)



with open("inference_timings.csv", 'w') as f:
    for model in models:
        data_row = []
        with open(f"./models/data/{model}", 'r') as file:
            csvreader = csv.reader(file, delimiter=' ')
            for row in csvreader:
                if len(row) != 0 and row[0] == 'Timings' and row[2] == 'count=50':
                    name = model.split('_')
                    model_name = name[1]+'_'+name[2]+'_'+name[3]+'_'+name[4].split('.')[0]
                    data_row.append(model_name)
                    data_row.append(row[6].split('=')[1])
                    # create the csv writer
                    writer = csv.writer(f)
                    # write a row to the csv file
                    writer.writerow(data_row)
