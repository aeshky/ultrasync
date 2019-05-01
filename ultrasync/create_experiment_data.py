"""

Date: Oct 2018
Author: Aciel Eshky

We have the lists of sample names in each subset (from create_train_test_splits).
This script uses the information from the list of samples to create batches for training, validation and testing.

"""

import os
import pandas as pd

from synchronisation.create_experiment_data_utils import create_data_file


def main():

    # path = '/afs/inf.ed.ac.uk/group/project/ultrax2020/aeshky/SyncData/'
    # dest_path = '/afs/inf.ed.ac.uk/group/project/ultrax2020/aeshky/experiments/sync_4'

    path = '/disk/scratch_big/aeshky/SyncDataSmallSilMfcc13/'
    dest_path = '/disk/scratch_big/aeshky/experiments/sync_10/'

    docs = os.path.join(dest_path, 'docs')

    df_train_shuffled = pd.read_csv(os.path.join(docs, "df_train_shuffled.csv"))
    df_val_1 = pd.read_csv(os.path.join(docs, "df_val_1.csv"))
    df_val_2 = pd.read_csv(os.path.join(docs, "df_val_2.csv"))
    df_val_3 = pd.read_csv(os.path.join(docs, "df_val_3.csv"))
    df_val_4 = pd.read_csv(os.path.join(docs, "df_val_4.csv"))
    df_val_5 = pd.read_csv(os.path.join(docs, "df_val_5.csv"))
    df_test_1 = pd.read_csv(os.path.join(docs, "df_test_1.csv"))
    df_test_2 = pd.read_csv(os.path.join(docs, "df_test_2.csv"))
    df_test_3 = pd.read_csv(os.path.join(docs, "df_test_3.csv"))
    df_test_4 = pd.read_csv(os.path.join(docs, "df_test_4.csv"))
    df_test_5 = pd.read_csv(os.path.join(docs, "df_test_5.csv"))

    # create train and test data
    num_samples_per_file = 64

    data = os.path.join(dest_path, 'data')
    if not os.path.exists(docs):
        os.makedirs(docs)

    if not os.path.exists(os.path.join(data, 'train')):
        os.makedirs(os.path.join(data, 'train'))

    if not os.path.exists(os.path.join(data, 'val')):
        os.makedirs(os.path.join(data, 'val'))

    if not os.path.exists(os.path.join(data, 'test')):
        os.makedirs(os.path.join(data, 'test'))

    create_data_file(path, os.path.join(data, 'train'), "train", num_samples_per_file, df_train_shuffled)
    create_data_file(path, os.path.join(data, 'val'), "val1", num_samples_per_file, df_val_1)
    create_data_file(path, os.path.join(data, 'val'), "val2", num_samples_per_file, df_val_2)
    create_data_file(path, os.path.join(data, 'val'), "val3", num_samples_per_file, df_val_3)
    create_data_file(path, os.path.join(data, 'val'), "val4", num_samples_per_file, df_val_4)
    create_data_file(path, os.path.join(data, 'val'), "val5", num_samples_per_file, df_val_5)
    create_data_file(path, os.path.join(data, 'test'), "test1", num_samples_per_file, df_test_1)
    create_data_file(path, os.path.join(data, 'test'), "test2", num_samples_per_file, df_test_2)
    create_data_file(path, os.path.join(data, 'test'), "test3", num_samples_per_file, df_test_3)
    create_data_file(path, os.path.join(data, 'test'), "test4", num_samples_per_file, df_test_4)
    create_data_file(path, os.path.join(data, 'test'), "test5", num_samples_per_file, df_test_5)


if __name__ == "__main__":
    main()
