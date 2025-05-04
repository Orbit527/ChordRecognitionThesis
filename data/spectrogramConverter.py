import os

import librosa
import matplotlib.pyplot as plt
import numpy as np

from LabelHelper import chordToLabel


def spectrogram_from_file(file, target):
    y, sr = librosa.load(file)
    diagram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, fmax=4000)

    plt.figure(figsize=(1, 1))
    librosa.display.specshow(librosa.amplitude_to_db(diagram, ref=np.max), sr=sr)

    plt.axis("off")

    plt.savefig(target, bbox_inches='tight', pad_inches=0, dpi=100)
    # plt.show()

    plt.close()
    print("converted: " + file)


def process_files(input_folder, output_folder):
    for file in os.listdir(input_folder):
        name_split = file.split("-")
        label = name_split[0]
        num = name_split[1].split(".")[0]

        file_path = input_folder + "/" + file
        target_path = output_folder + "/" + str(chordToLabel[label]) + "_" + label + "_" + num + ".png"

        spectrogram_from_file(file_path, target_path)


if __name__ == "__main__":
    inputFolder = "./rawHigher"
    outputFolder = "./spectrogramHigher"

    process_files(inputFolder, outputFolder)
    # spectrogram_from_fileScreenshotDELETE("./rawHigher/A-01.wav", "test")
