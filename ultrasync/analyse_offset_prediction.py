"""

Date: Jan 2019
Author: Aciel Eshky

A script that analyses the offset prediction across different attributes and produces the tables for the
2019 interspeech paper.

"""


import os
import pandas as pd


def percentage(value):
    if not value:
        return None
    return str(round(value * 100, 1)) + "%"


def get_subset_type(file_name):

    unseen_speaker_subsets = ["test_1", "test_3", "test_5"]
    unseen_session_subsets = ["test_2", "test_4"]

    # unseen_speaker_subsets = ["val1", "val2", "val4", "val6"]
    # unseen_session_subsets = ["val3", "val5"]

    if any(test_set in file_name for test_set in unseen_speaker_subsets):
        return "unseen_speaker"
    elif any(test_set in file_name for test_set in unseen_session_subsets):
        return "unseen_session"
    else:
        return None


def get_results_per_subset_per_type(df, column, dataset, subset, types=("A", "B", "C", "D", "F")):
    if subset is None:
        results = [dataset]
    else:
        results = [dataset, subset]

    for t in types:
        s = df[df.type == t]
        results.append(s.shape[0])
        results.append(percentage(s[column].mean()))
    return results


def get_table_per_subset_per_type(df, column):

    uxtd = df[df.dataset == "uxtd"]
    uxssd_unseen_session = df[(df.dataset == "uxssd") & (df.subset == "unseen_session")]
    uxssd_unseen_speaker = df[(df.dataset == "uxssd") & (df.subset == "unseen_speaker")]
    upx_unseen_session = df[(df.dataset == "upx") & (df.subset == "unseen_session")]
    upx_unseen_speaker = df[(df.dataset == "upx") & (df.subset == "unseen_speaker")]

    table = [
        ['', '',
         'Words (A)', '',
         'Non-words (B)', '',
         'Sentence (C)', '',
         'Articulatory (D)', '',
         'Conversation (F)', ''],
        ['Dataset', 'Subset', 'N', 'Accuracy', 'N', 'Accuracy', 'N', 'Accuracy', 'N', 'Accuracy', 'N', 'Accuracy'],
        get_results_per_subset_per_type(uxtd, dataset='UXTD', subset='new speakers', column=column),
        get_results_per_subset_per_type(uxssd_unseen_session, dataset='UXSSD', subset='new sessions', column=column),
        get_results_per_subset_per_type(uxssd_unseen_speaker, dataset='UXSSD', subset='new speaker', column=column),
        get_results_per_subset_per_type(upx_unseen_session, dataset='UPX', subset='new sessions', column=column),
        get_results_per_subset_per_type(upx_unseen_speaker, dataset='UPX', subset='new speaker', column=column),
        get_results_per_subset_per_type(df, dataset='All', subset='', column=column)]
    return pd.DataFrame(table)


def get_table_per_subset(df, column):

    uxtd = df[df.dataset == "uxtd"]
    uxssd_unseen_session = df[(df.dataset == "uxssd") & (df.subset == "unseen_session")]
    uxssd_unseen_speaker = df[(df.dataset == "uxssd") & (df.subset == "unseen_speaker")]
    upx_unseen_session = df[(df.dataset == "upx") & (df.subset == "unseen_session")]
    upx_unseen_speaker = df[(df.dataset == "upx") & (df.subset == "unseen_speaker")]

    cols = ['Dataset', 'Subset', 'N', 'Accuracy', 'Discrepancy_Mean', 'Discrepancy_SD']
    table = [
        ['UXTD', 'new speakers',
         uxtd.shape[0],
         uxtd[column].mean(),
         uxtd.difference.mean(),
         uxtd.difference.std()],

        ['UXSSD', 'new sessions',
         uxssd_unseen_session.shape[0],
         uxssd_unseen_session[column].mean(),
         uxssd_unseen_session.difference.mean(),
         uxssd_unseen_session.difference.std()],

        ['UXSSD', 'new speaker',
         uxssd_unseen_speaker.shape[0],
         uxssd_unseen_speaker[column].mean(),
         uxssd_unseen_speaker.difference.mean(),
         uxssd_unseen_speaker.difference.std()],

        ['UPX', 'new sessions',
         upx_unseen_session.shape[0],
         upx_unseen_session[column].mean(),
         upx_unseen_session.difference.mean(),
         upx_unseen_session.difference.std()],

        ['UPX', 'new speaker',
         upx_unseen_speaker.shape[0],
         upx_unseen_speaker[column].mean(),
         upx_unseen_speaker.difference.mean(),
         upx_unseen_speaker.difference.std()],

        ['All', '',
         df.shape[0],
         df[column].mean(),
         df.difference.mean(),
         df.difference.std()]
    ]
    table = pd.DataFrame(table, columns=cols)
    table['Accuracy'] = table['Accuracy'].apply(lambda x: percentage(x))
    table['Discrepancy_Mean'] = table['Discrepancy_Mean'].apply(lambda x: round(x))
    table['Discrepancy_SD'] = table['Discrepancy_SD'].apply(lambda x: round(x))

    return table


def get_table_per_dataset_per_type(df, column):

    uxtd = df[df.dataset == "uxtd"]
    uxssd = df[df.dataset == "uxssd"]
    upx = df[df.dataset == "upx"]

    table = [
        ['',
         'Words (A)', '',
         'Non-words (B)', '',
         'Sentence (C)', '',
         'Articulatory (D)', '',
         'Conversation (F)', ''],
        ['Dataset', 'N', 'Accuracy', 'N', 'Accuracy', 'N', 'Accuracy', 'N', 'Accuracy', 'N', 'Accuracy'],
        get_results_per_subset_per_type(uxtd, dataset='UXTD', subset=None, column=column),
        get_results_per_subset_per_type(uxssd, dataset='UXSSD', subset=None, column=column),
        get_results_per_subset_per_type(upx, dataset='UPX', subset=None, column=column),
        get_results_per_subset_per_type(df, dataset='All', subset=None, column=column)]
    return pd.DataFrame(table)


def get_table_per_type(df, column):

    table = [
        ['Words (A)', df[df.type == "A"].shape[0], df[df.type == "A"][column].mean(),
         df[df.type == "A"]['difference'].mean(),
         df[df.type == "A"]['difference'].std()],
        ['Non-words (B)', df[df.type == "B"].shape[0], df[df.type == "B"][column].mean(),
         df[df.type == "B"]['difference'].mean(),
         df[df.type == "B"]['difference'].std()],
        ['Sentence (C)', df[df.type == "C"].shape[0], df[df.type == "C"][column].mean(),
         df[df.type == "C"]['difference'].mean(),
         df[df.type == "C"]['difference'].std()],
        ['Articulatory (D)', df[df.type == "D"].shape[0], df[df.type == "D"][column].mean(),
         df[df.type == "D"]['difference'].mean(),
         df[df.type == "D"]['difference'].std()],
        ['Conversation (F)', df[df.type == "F"].shape[0], df[df.type == "F"][column].mean(),
         df[df.type == "F"]['difference'].mean(),
         df[df.type == "F"]['difference'].std()],
        ['All', df.shape[0], df[column].mean(),
         df['difference'].mean(),
         df['difference'].std()],
        ['A, B, C and F',
         df[df.type.isin(['A', 'B', 'C', 'F'])].shape[0], df[df.type.isin(['A', 'C', 'F'])][column].mean(),
         df[df.type.isin(['A', 'C', 'F'])]['difference'].mean(),
         df[df.type.isin(['A', 'C', 'F'])]['difference'].std()]
    ]
    table = pd.DataFrame(table, columns=["Utterance Type", "N", "Accuracy",
                                         "Discrepancy_Mean", "Discrepancy_SD"])
    table['Accuracy'] = table['Accuracy'].apply(lambda x: percentage(x))
    table['Discrepancy_Mean'] = table['Discrepancy_Mean'].apply(lambda x: round(x))
    table['Discrepancy_SD'] = table['Discrepancy_SD'].apply(lambda x: round(x))
    return table


def get_mean_discrepancy(difference):

    positive_diff = difference[difference >= 0]
    negative_diff = difference[difference < 0]

    print(positive_diff.describe())
    print("median\t", positive_diff.median())
    print("\n")
    print(negative_diff.describe())
    print("median\t", negative_diff.median())
    print("\n")
    print(abs(difference).describe())
    print("median\t", difference.median())
    print("\n")
    print(difference.describe())
    print("median\t", difference.median())


def main():

    path = "/Users/acieleshky/Documents/synchronisation/sync_10/output/predictions/20190207-09hr30m13s/predictions"

    df = pd.DataFrame()

    evaluation_column = "detectability_correct"

    file_names = [f for f in os.listdir(path) if f.startswith("offset_prediction")]

    for file in file_names:

        if file.endswith(".csv"):

            temp_df = pd.read_csv(os.path.join(path, file))
            temp_df['subset'] = get_subset_type(file)

            df = pd.concat([df, temp_df])

    # mutiply the difference by 1000 to convert from seconds to miliseconds and make it consistent with the
    # rest of the paper
    df['difference'] = df.apply(lambda row: (row['offset_prediction'] - row['offset_true']) * 1000, axis=1)

    # extract the type of utterance from the file name
    df['type'] = df['base_file_name'].apply(lambda x: list(str(x))[-1])
    df = df[df.type != "E"]  # remove utterance type "other" (E) which include coughs and swallows.
    df = df.reset_index()

    # apply the detectability threshold
    df["detectability_correct"] = df.difference.apply(lambda x: 1 if (x < 45) and (x > -125) else 0)

    # overall accuracy
    print("Percentage of utterances with a correct offset prediction (accuracy): " +
          str(percentage(df.correct.mean())) + "%")

    # create the results tables
    table_1 = get_table_per_subset(df, column=evaluation_column)
    table_2 = get_table_per_type(df, column=evaluation_column)
    table_3 = get_table_per_subset_per_type(df, column=evaluation_column)
    table_4 = get_table_per_dataset_per_type(df, column=evaluation_column)

    print(table_1)
    print(table_2)
    print(table_3)
    print(table_4)

    out_path = os.path.join(path, "new_out")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    pd.DataFrame.to_csv(table_1, os.path.join(out_path, "table1.csv"), index=False)
    pd.DataFrame.to_csv(table_2, os.path.join(out_path, "table2.csv"), index=False)
    pd.DataFrame.to_csv(table_3, os.path.join(out_path, "table3.csv"), index=False, header=False)
    pd.DataFrame.to_csv(table_4, os.path.join(out_path, "table4.csv"), index=False, header=False)

    # these are easier to format for latex
    pd.DataFrame.to_csv(table_1, os.path.join(out_path, "table1.txt"), index=False, sep="&")
    pd.DataFrame.to_csv(table_2, os.path.join(out_path, "table2.txt"), index=False, sep="&")
    pd.DataFrame.to_csv(table_3, os.path.join(out_path, "table3.txt"), index=False, header=False, sep="&")
    pd.DataFrame.to_csv(table_4, os.path.join(out_path, "table4.txt"), index=False, header=False, sep="&")

    print(get_mean_discrepancy(df.difference))


if __name__ == "__main__":
    main()
