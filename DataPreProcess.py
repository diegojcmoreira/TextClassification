import os
from os import listdir
from os.path import isfile, join
import pandas as pd

from nltk.corpus import stopwords

SEPARADOR = ' '


def build_csv_with_data(working_folder, dataset_path, filename):
    all_folders = [element for element in listdir(dataset_path) if not isfile(join(dataset_path, element))]
    output_file_name = os.path.join(working_folder, filename)
    print(output_file_name)

    with open(output_file_name, 'w') as outputFile:
        for folder in all_folders:
            print(folder)
            folder_path = os.path.join(dataset_path, folder)
            all_files = [f for f in listdir(folder_path) if isfile(os.path.join(folder_path, f))]
            for file in all_files:
                text = ""
                with open(os.path.join(folder_path, file), encoding="utf8") as infile:
                    # firstLine = infile.readline().replace("\n","")
                    for line in infile:
                        # remove stopwords do texto de concatena as linhas para formar um CSV
                        filtered_words = [word for word in line.split(' ') if word not in stopwords.words('portuguese')]
                        line = SEPARADOR.join(filtered_words)
                        text = text + " " + line.replace("\n", "")
                outputFile.write(folder + "\t" + text + "\n")

    return output_file_name


def read_data(filepath):
    data_frame = pd.read_csv(filepath, encoding="ISO-8859-1", sep='\t', header=None, skiprows=0,
                             names=["Class", "Text"])

    return data_frame.groupby('Class').filter(lambda x: len(x) > 10)
