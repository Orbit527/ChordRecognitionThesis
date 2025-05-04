import csv
import os

import librosa
import numpy as np

from LabelHelper import *


def file_to_pcp(file):
    y, sr = librosa.load(file)
    pcp_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
    # pcpSq = librosa.feature.chroma_cqt(y=y, sr=sr)

    arr = []

    for x in pcp_stft:
        arr.append(np.round(np.mean(x), 4))

    print(arr)
    return arr


def process_files(input_folder, output_folder, sample_rate=22050):
    output_file_name = "pcpData.csv"
    # clear output File
    open(output_folder + '/' + output_file_name, 'w').close()

    with open(output_folder + '/' + output_file_name, 'w', newline='') as f:
        csv_writer = csv.writer(f)

        for file in os.listdir(input_folder):
            label = chordToLabel[file.split("-")[0]]

            pcp_vector = file_to_pcp(input_folder + '/' + file)
            arr = np.asarray(pcp_vector)
            arr_with_label = np.insert(arr, 0, label)
            csv_writer.writerow(arr_with_label)

            print('File: ' + file + ' with label ' + str(label) + ' converted to chroma')


if __name__ == '__main__':
    inputFolder = './rawHigher'
    outputFolder = './pcpHigher'

    process_files(inputFolder, outputFolder)
