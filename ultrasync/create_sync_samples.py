"""

Date: Jul 2018
Author: Aciel Eshky


A script to create positive and negative samples using self-supervision.

"""

import os
import sys
import random
import pandas as pd
from numpy.random import seed as np_seed

from ustools.folder_utils import get_utterance_id, get_dir_info
from synchronisation.create_sync_samples_utils import create_samples, save_samples_to_disk

random.seed(2018)
np_seed(2018)


def mirror_folder_structure(input_path, output_path):
    """
    Function to create a mirror of the input path folder structure in output path.
    Adapted from https://stackoverflow.com/a/40829525/5190279

    :param input_path:
    :param output_path:
    :return: a list of pairs of core dir which contains files, and corresponding generated dir
    """

    folder_pairs = []

    for dirpath, dirnames, filenames in os.walk(input_path):

        dirnames.sort()

        if any(fname.endswith('.ult') for fname in filenames):

            new_output_folder = os.path.join(output_path, dirpath[len(input_path):])

            if not os.path.isdir(new_output_folder):
                print("Creating folder: \t" + new_output_folder)
                os.makedirs(new_output_folder)
            else:
                print("Folder already exits: \t" + new_output_folder)

            if filenames:  # return the dirs that contain files
                folder_pairs.append([dirpath, new_output_folder])

    return folder_pairs


def get_file_basenames(directory):
    files = os.listdir(directory)
    return set([f.split('.')[0] for f in files])


def create_sync_data(folder_pairs):

    files_created = []

    df = pd.DataFrame(folder_pairs, columns=("core", "generated"))

    for index, row in df.iterrows(): # itertools.islice(df.iterrows(), 80):

        print("Processing: ", row['core'], row['generated'])
        core_dir = row['core']
        target_dir = row['generated']

        basenames = get_file_basenames(core_dir)
        target_basenames = get_file_basenames(target_dir)

        for b in basenames:

            # if os.path.isfile(os.path.join(target_dir, b + '.npz')):
            #     print(os.path.join(target_dir, b + '.npz'), "already exists.")

            if [b in i for i in target_basenames]:
                print(b, "files already exist in target directory.")

            elif "E" in b:
                print("Skipping utterance of type \"non-speech\" (E):", os.path.join(target_dir, b))

            else:
                try:

                    info = get_dir_info(core_dir)
                    root_id = get_utterance_id(info['dataset'], info['speaker'], info['session'], b)
                    print(root_id)

                    samples = create_samples(core_dir, b)

                    chunk_names = save_samples_to_disk(samples, root_id, target_dir)

                    list.extend(files_created, chunk_names)

                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    print("not_processed: ", core_dir, b)

    return files_created


def main():

    ultrasuite = ["uxtd", "uxssd", "upx"]

    # inputpath = sys.argv[1]
    # outputpath = sys.argv[2]

    # the location of the original ultrasuite data
    input_path = "/group/project/ultrax2020/UltraSuite/"

    # the destination: where the sync dataset will be stored.
    # This will consists of of samples, each corresponding to 200 ms of ultrasound and audio.
    output_path = "/disk/scratch_big/aeshky/SyncDataSmallSilMfcc13"

    for dataset in ultrasuite:

        docs = os.path.join(output_path, "docs", dataset)
        if not os.path.exists(docs):
            os.makedirs(docs)

        input_path_data = os.path.join(input_path, "core-" + dataset, "core/")  # this slash is very important!
        output_path_data = os.path.join(output_path, dataset)

        print("processing", dataset,
              "input directory is:", input_path_data,
              "output directory is:", output_path_data)

        # source and destination folder pairs.
        # destination is created by mirror source.
        folder_pairs = mirror_folder_structure(input_path_data, output_path_data)

        # save the pairs for logging purposes
        pd.DataFrame.to_csv(pd.DataFrame(columns={"source", "target"}, data=folder_pairs),
                            os.path.join(docs, "folder_pairs.csv"), index=False)

        # create and save the data
        file_names = create_sync_data(folder_pairs)

        # save the sample file names for logging purposes
        pd.DataFrame.to_csv(pd.DataFrame(columns={"file_names"}, data=file_names),
                            os.path.join(docs, "file_names.csv"), index=False)


if __name__ == "__main__":
    main()
