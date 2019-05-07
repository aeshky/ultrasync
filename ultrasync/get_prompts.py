"""

Date: Mar 2019
Author: Aciel Eshky

A script that get the prompts of the utterances for further analysis

"""

import os
import pandas as pd
import io


def read_prompt(file):
    """
    Read prompt file containing prompt, datetime, and speaker ID
    :param file:
    :return:
    """
    with io.open(file, mode="r", encoding='utf-8', errors='ignore') as prompt_f:
        return prompt_f.readline().rstrip()


def main():
    experiment_id = "20190207-09hr30m13s"
    file_prefix = "offset_predictions"
    path = "/disk/scratch_big/../experiments/sync_10/output/predictions/"
    out_path = "/disk/scratch_big/../experiments/sync_10/docs/prompts"

    files = [i for i in os.listdir(path) if i.startswith(file_prefix) and experiment_id in i]

    for f in files:
        df = pd.read_csv(os.path.join(path, f))
        df['prompt_file'] = df.apply(lambda row: os.path.join(row['path'], row['base_file_name'] + '.txt'), axis=1)
        df['prompt'] = df.apply(lambda row: read_prompt(row['prompt_file']), axis=1)
        test_file_id = f.split(file_prefix)[1].split(experiment_id)[0]
        df = df.drop(['prompt_file'], axis=1)
        pd.DataFrame.to_csv(df, os.path.join(out_path, "df" + test_file_id + "prompt.csv"))


if __name__ == "__main__":
    main()


