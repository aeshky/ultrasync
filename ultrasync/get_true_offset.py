"""

Date: Oct 2018
Author: Aciel Eshky

Get the true offsets of the train val and test data. The true offsets of test data are used as
ground truth for evaluation.

"""

import os
import pandas as pd
import numpy as np
from synchronisation.create_train_test_splits import get_train_val_test_splits
from ustools.folder_utils import get_dir_info


def prepare_df(params_df):

    params_df = params_df[["FileName", "TimeInSecsOfFirstFrame"]]

    params_df = params_df.rename(index=str, columns={"TimeInSecsOfFirstFrame": "offset_true"})

    # split the path and base file name

    params_df["path"] = params_df.FileName.apply(lambda x: '/'.join(x.split("/")[:-1]))

    params_df["base_file_name"] = params_df.FileName.apply(lambda x: (x.split("/")[-1]).split(".")[0])

    columns = ['path', 'base_file_name', 'offset_true']

    params_df = params_df[columns]

    return params_df


def main():

    docs = '/disk/scratch_big/aeshky/experiments/sync_10/docs'
    out_docs = '/disk/scratch_big/aeshky/experiments/sync_10/docs/offset_true'
    # docs = '~/Documents/params_df/'

    bin_size = 0.045

    uxtd = pd.read_csv(os.path.join(docs, 'df_params_uxtd.csv'))
    uxssd = pd.read_csv(os.path.join(docs, 'df_params_uxssd.csv'))
    upx = pd.read_csv(os.path.join(docs, 'df_params_upx.csv'))

    params_df = pd.concat([uxtd, uxssd, upx])
    params_df = params_df.reset_index()

    params_df = prepare_df(params_df)

    params_df['dataset'] = ''
    params_df['speaker'] = ''
    params_df['session'] = ''

    for index, row in params_df.iterrows():
        info_dict = get_dir_info(row['path'])
        params_df.at[index, 'dataset'] = info_dict['dataset']
        params_df.at[index, 'speaker'] = info_dict['speaker']
        params_df.at[index, 'session'] = info_dict['session']

    # get utterance type
    params_df['type'] = params_df['base_file_name'].apply(lambda x: list(str(x))[-1])
    params_df = params_df[params_df.type != 'E']  # remove utterance type "other" (E) which include coughs and swallows.
    params_df = params_df.reset_index()

    # get true offset bin
    min_val = min(params_df["offset_true"])
    max_val = max(params_df["offset_true"])
    print("min val:", min_val, "- max val:", max_val)
    candidates = np.round(np.arange(min_val, max_val + bin_size, bin_size), 3)
    params_df['ind'] = np.digitize(params_df["offset_true"], candidates)
    params_df['offset_true_bin'] = params_df["ind"].apply(lambda x: candidates[x - 1])
    params_df.drop(['ind'], axis=1)

    # column names
    columns = ['path', 'base_file_name', 'type', 'dataset', 'speaker', 'session', 'offset_true', 'offset_true_bin']

    [df_train_1, df_train_2, df_train_3,
     df_val_1, df_val_2, df_val_3, df_val_4, df_val_5,
     df_test_1, df_test_2, df_test_3, df_test_4, df_test_5] = get_train_val_test_splits(params_df)

    if not os.path.exists(out_docs):
        os.mkdir(out_docs)

    pd.DataFrame.to_csv(df_train_1, os.path.join(out_docs, "df_offset_train_1.csv"), index=False, columns=columns)
    pd.DataFrame.to_csv(df_train_2, os.path.join(out_docs, "df_offset_train_2.csv"), index=False, columns=columns)
    pd.DataFrame.to_csv(df_train_3, os.path.join(out_docs, "df_offset_train_3.csv"), index=False, columns=columns)

    pd.DataFrame.to_csv(df_val_1, os.path.join(out_docs, "df_offset_val_1.csv"), index=False, columns=columns)
    pd.DataFrame.to_csv(df_val_2, os.path.join(out_docs, "df_offset_val_2.csv"), index=False, columns=columns)
    pd.DataFrame.to_csv(df_val_3, os.path.join(out_docs, "df_offset_val_3.csv"), index=False, columns=columns)
    pd.DataFrame.to_csv(df_val_4, os.path.join(out_docs, "df_offset_val_4.csv"), index=False, columns=columns)
    pd.DataFrame.to_csv(df_val_5, os.path.join(out_docs, "df_offset_val_5.csv"), index=False, columns=columns)

    pd.DataFrame.to_csv(df_test_1, os.path.join(out_docs, "df_offset_test_1.csv"), index=False, columns=columns)
    pd.DataFrame.to_csv(df_test_2, os.path.join(out_docs, "df_offset_test_2.csv"), index=False, columns=columns)
    pd.DataFrame.to_csv(df_test_3, os.path.join(out_docs, "df_offset_test_3.csv"), index=False, columns=columns)
    pd.DataFrame.to_csv(df_test_4, os.path.join(out_docs, "df_offset_test_4.csv"), index=False, columns=columns)
    pd.DataFrame.to_csv(df_test_5, os.path.join(out_docs, "df_offset_test_5.csv"), index=False, columns=columns)


if __name__ == "__main__":
    main()
