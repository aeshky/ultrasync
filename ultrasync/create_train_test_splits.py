"""

Date: Oct 2018
Author: Aciel Eshky

This is a script that outputs a list of names of samples for training development and test sets.
This list is then used to create batches for training validation and testing.
The speakers / session names in each subset have been hard-coded.
The sample names are listed and in the case of training data are randomised.

"""

import os
import sys
import random
import pandas as pd
import numpy as np

from ultrasync.create_experiment_data_utils import get_sync_file_names, split_name

random.seed(2018)
np.random.seed(2018)


def split_val_and_test(df):
    half = len(df)//2
    df1 = df.iloc[:half, :]
    df2 = df.iloc[half:, :]
    return df1, df2


def get_train_val_test_splits(df_info):
    # upx
    # the training data excludes speakers 01F and 15M and session BL3
    df_train_1 = df_info[(df_info.dataset == "upx") &
                         (df_info.speaker != '01F') & (df_info.speaker != '15M') &
                         (df_info.session != 'BL3')]

    # there are two validation sets: one with all the data of one held out speaker (01F)
    # and the other with a held out session from first half of the remaining speakers (excluding 01F, of course)
    # ~ 11% of upx
    df_val_1 = df_info[(df_info.dataset == "upx") & (df_info.speaker == '01F')]  # 6% of the data
    df_val_2 = df_info[(df_info.dataset == "upx") & (df_info.session == 'BL3') &
                       df_info.speaker.isin(["02F", "03F", "04M", "05M", "06M", "07M", "08M", "09M", "10M"])]  # 5%

    # there are two test sets: one with all the data of one held out speaker (15M)
    # and the other with a held out session from second half of the remaining speakers (excluding 15M, of course)
    # ~ 9% of upx
    df_test_1 = df_info[(df_info.dataset == "upx") & (df_info.speaker == '15M')]  # 5% of the data
    df_test_2 = df_info[(df_info.dataset == "upx") & (df_info.session == 'BL3') &
                        df_info.speaker.isin(["11M", "12M", "13M", "14M", "16M", "17M", "18F", "19M", "20M"])]  # 4%

    # uxssd
    # the training data excludes speakers 01M and 07F and session Mid
    df_train_2 = df_info[(df_info.dataset == "uxssd") &
                         (df_info.speaker != '01M') & (df_info.speaker != '07F') &
                         (df_info.session != 'Mid')]

    # there are two validation sets: one with all the data of one held out speaker (01M)
    # and the other with a held out session from first half of the remaining speakers (excluding 01M, of course)
    # 11%
    df_val_3 = df_info[(df_info.dataset == "uxssd") & (df_info.speaker == '01M')]
    df_val_4 = df_info[(df_info.dataset == "uxssd") & (df_info.session == 'Mid') &
                       df_info.speaker.isin(['02M', '03F', '04M'])]

    # there are two test sets: one with all the data of one held out speaker (07F)
    # and the other with a held out session from second half of the remaining speakers (excluding 15M, of course)
    # 11%
    df_test_3 = df_info[(df_info.dataset == "uxssd") & (df_info.speaker == '07F')]
    df_test_4 = df_info[(df_info.dataset == "uxssd") & (df_info.session == 'Mid') &
                        df_info.speaker.isin(['05M', '06M', '08M'])]

    # uxtd
    # we hold out some speakers which make up 20% of the data
    uxtd_val_speakers = ["07F", "08M", "12M", "13F", "26F"]  # val speakers 10% of uxtd
    uxtd_test_speakers = ["30F", "38M", "43F", "45M", "47M", "52F", "53F", "55M"]  # test speakers 10% of uxtd
    # 80 %
    df_train_3 = df_info[(df_info.dataset == "uxtd") &
                         (~df_info.speaker.isin(uxtd_val_speakers)) &
                         (~df_info.speaker.isin(uxtd_test_speakers))]
    # 10%
    df_val_5 = df_info[(df_info.dataset == "uxtd") & (df_info.speaker.isin(uxtd_val_speakers))]
    # 10%
    df_test_5 = df_info[(df_info.dataset == "uxtd") & (df_info.speaker.isin(uxtd_test_speakers))]

    splits = [df_train_1, df_train_2, df_train_3,
              df_val_1, df_val_2, df_val_3, df_val_4, df_val_5,
              df_test_1, df_test_2, df_test_3, df_test_4, df_test_5]

    return splits


def main():

    path = sys.argv[1]  # '/disk/scratch_big/../SyncDataSmall../'
    dest_path = sys.argv[2]  # '/disk/scratch_big/../experiments/sync_10/'

    files = get_sync_file_names(path)
    df_info = pd.DataFrame(data=[split_name(i) for i in files])
    col_order = ['filename', 'dataset', 'speaker', 'session', 'utterance', 'chunk']
    df_info = df_info[col_order]
    df_info = df_info.sort_values(by="filename")

    docs = os.path.join(dest_path, 'docs')
    if not os.path.exists(docs):
        os.makedirs(docs)

    pd.DataFrame.to_csv(df_info, os.path.join(docs, 'file_names.csv'), index=False)

    [df_train_1, df_train_2, df_train_3,
     df_val_1, df_val_2, df_val_3, df_val_4, df_val_5,
     df_test_1, df_test_2, df_test_3, df_test_4, df_test_5] = get_train_val_test_splits(df_info)

    # save the splits
    pd.DataFrame.to_csv(df_train_1, os.path.join(docs, "df_train_1.csv"), index=False)
    pd.DataFrame.to_csv(df_train_2, os.path.join(docs, "df_train_2.csv"), index=False)
    pd.DataFrame.to_csv(df_train_3, os.path.join(docs, "df_train_3.csv"), index=False)

    pd.DataFrame.to_csv(df_val_1, os.path.join(docs, "df_val_1.csv"), index=False)
    pd.DataFrame.to_csv(df_val_2, os.path.join(docs, "df_val_2.csv"), index=False)
    pd.DataFrame.to_csv(df_val_3, os.path.join(docs, "df_val_3.csv"), index=False)
    pd.DataFrame.to_csv(df_val_4, os.path.join(docs, "df_val_4.csv"), index=False)
    pd.DataFrame.to_csv(df_val_5, os.path.join(docs, "df_val_5.csv"), index=False)

    pd.DataFrame.to_csv(df_test_1, os.path.join(docs, "df_test_1.csv"), index=False)
    pd.DataFrame.to_csv(df_test_2, os.path.join(docs, "df_test_2.csv"), index=False)
    pd.DataFrame.to_csv(df_test_3, os.path.join(docs, "df_test_3.csv"), index=False)
    pd.DataFrame.to_csv(df_test_4, os.path.join(docs, "df_test_4.csv"), index=False)
    pd.DataFrame.to_csv(df_test_5, os.path.join(docs, "df_test_5.csv"), index=False)

    df_train = pd.concat([df_train_1, df_train_2, df_train_3])

    # shuffle training data

    stem = list(set(df_train.filename.apply(lambda x: '_'.join(x.split("_")[:-1]))))
    np.random.shuffle(stem)
    shuffled_files = []
    for s in stem:
        shuffled_files.append(s + '_neg')
        shuffled_files.append(s + '_pos')

    assert len(shuffled_files) == len(df_train)

    df_train_shuffled = pd.DataFrame(data=[split_name(i) for i in shuffled_files])

    pd.DataFrame.to_csv(df_train_shuffled, os.path.join(docs, "df_train_shuffled.csv"), index=False)


if __name__ == "__main__":
    main()
