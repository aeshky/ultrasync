"""

Date: Oct 2018
Author: Aciel Eshky

Utilities for create sync experiment data.

"""

import os
import numpy as np


def get_sync_file_names(path):
    files = []
    for dp, dn, filenames in os.walk(path):
        for f in filenames:
            f = f.split('.')
            if f[1] == 'npz':
                print(f[0])
                files.append(f[0])
    return files


def split_name(fname):
    l = fname.split("-")
    return {"filename": fname,
            "dataset": l[0],
            "speaker": l[1],
            "session": l[2],
            "utterance": l[3],
            "chunk": l[4]}


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def create_data_file(source_path, dest_path, filename_prefix, num_samples_per_file, df_info):
    """
    Iterates over the names in df_info to create batches of samples

    :param source_path:
    :param dest_path:
    :param filename_prefix:
    :param num_samples_per_file: size of batch
    :param df_info: contains a list of sample names
    :return:
    """
    df_info_chunked = list(chunks(df_info, num_samples_per_file))

    for i, ch in enumerate(df_info_chunked):

        file = os.path.join(dest_path, filename_prefix + "-" + str(i))

        if os.path.exists(file + ".npz"):
            print(file, "already exists. Skipping to next file.")

        else:

            print(i, len(ch), len(ch) == num_samples_per_file)

            samples = {"name": [],
                       "label": [],
                       "ult": [],
                       "mfcc": []}

            if len(ch) == num_samples_per_file:

                ch = ch.reset_index(drop=True)

                for index, row in ch.iterrows():
                    # print(index, row["filename"])

                    if row["session"] == "Single":
                        sample = np.load(os.path.join(source_path,
                                                      row["dataset"],
                                                      row["speaker"],
                                                      row["filename"] + ".npz"))
                    else:
                        sample = np.load(os.path.join(source_path,
                                                      row["dataset"],
                                                      row["speaker"],
                                                      row["session"],
                                                      row["filename"] + ".npz"))

                    samples['name'].append(row["filename"])
                    samples['label'].append(sample['label'])
                    samples['ult'].append(sample['ult'])
                    samples['mfcc'].append(sample['mfcc'])

                    assert len(samples['name']) == len(samples['label']) == len(samples['ult']) == len(samples['mfcc'])

                print("Saving to " + file + "...")

                np.savez(file,
                         name=samples['name'],
                         label=samples['label'],
                         ult=samples['ult'],
                         mfcc=samples['mfcc'])
