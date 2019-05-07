"""

Date: Nov 2018
Author: Aciel Eshky

Retrieves sample counts to help select train, validation and testing subsets.

We have already created sync samples using script "create_sync_samples".
This script gets the numbers of samples for each datasets, speakers, and sessions.
These counts are used to select training, validation and test subsets.
These are chosen randomly based only on the relative sizes of subsets.
We end up with ~80% training, 10% validation and 10% testing (in terms of number of samples, which are 200 ms each).

"""

import pandas as pd

from synchronisation.create_experiment_data_utils import get_sync_file_names, split_name


def main():

    PATH = "/disk/scratch_big/../SyncDataSmallSil/"

    output_file_names = "/disk/scratch_big/../SyncDataSmallSil/docs/df_file_names.csv"
    output_file_info = "/disk/scratch_big/../SyncDataSmallSil/docs/df_file_info.csv"
    output_speakers = "/disk/scratch_big/../SyncDataSmallSil/docs/df_speakers.csv"

    print("finding .npz files in ", PATH)
    files = get_sync_file_names(PATH)

    files.sort()

    # file names
    df_files = pd.DataFrame(columns={"filename"}, data=files)
    pd.DataFrame.to_csv(df_files, output_file_names, index=False)

    # file info
    df_info = pd.DataFrame(data=[split_name(i) for i in df_files.filename])
    col_order = ["filename", "dataset", "speaker", "session", "utterance", "chunk"]
    df_info = df_info[col_order]
    pd.DataFrame.to_csv(df_info, output_file_info, index=False)

    # speaker info
    df_speakers = df_info.drop(['chunk', 'filename', 'utterance'], axis=1)
    speaker_size = df_speakers.groupby(['speaker', 'dataset']).size().reset_index(name='counts')
    df_speakers = pd.DataFrame(speaker_size)
    pd.DataFrame.to_csv(df_speakers, output_speakers, index=False)


if __name__ == "__main__":
    main()
