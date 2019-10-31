"""

Date: Oct 2019
Author: Aciel Eshky

A script to analyse the offset prediction of articulatory utterances.

"""


import os
import sys
import pandas as pd


def analyse_stops(df_articulatory):

    df_stop_aca = df_articulatory[(df_articulatory['prompt'].str.contains("stop")) |
                                  (df_articulatory['prompt'].str.contains("aCa"))]

    total = len(df_stop_aca)

    correct = len(df_stop_aca[df_stop_aca['detectability_correct'] == 1])

    return total, correct, round(correct / total * 100, 1)


def analyse_vowels(df_articulatory):

    df_vowel = df_articulatory[df_articulatory['prompt'].str.contains("vowel")]

    total = len(df_vowel)

    correct = len(df_vowel[df_vowel["detectability_correct"] == 1])

    return total, correct, round(correct / total * 100, 1)


def main():

    experiment_id = "20190207-09hr30m13s"
    path = sys.argv[1]  # path to output e.g., "../ultrasync/output
    path_to_predictions = os.path.join(path, "predictions", "ultrasuite")
    path_to_prompts = sys.argv[2]  # ../ultrasync/docs/prompts"

    expected_columns = ["path", "base_file_name", "type", "dataset", "speaker", "session", "prompt"]

    # read predictions
    df_prediction = pd.read_csv(os.path.join(path_to_predictions, "full_predictions_" + experiment_id + ".csv"))
    print("df_predictions shape:", df_prediction.shape)
    print("df_predictions columns:", df_prediction.columns)
    assert(len(df_prediction) == 1502)

    # read prompts
    prompt_files = [os.path.join(path_to_prompts, f)
                    for f in os.listdir(path_to_prompts) if f.startswith("df_test_") and f.endswith("prompt.csv")]

    df_prompts = pd.DataFrame()
    for f in prompt_files:
        temp_df = pd.read_csv(f)
        temp_df = temp_df[expected_columns]
        df_prompts = pd.concat([df_prompts, temp_df])
    print("\ndf_prompt shape:", df_prompts.shape)
    print("df_prompt columns:", df_prompts.columns)
    assert(len(df_prediction) == len(df_prompts))

    # combine
    expected_columns.remove("prompt")
    df = pd.merge(df_prompts, df_prediction, on=expected_columns)

    print("\nmerged dataframe shape:", df.shape)
    assert(len(df) == len(df_prediction) == len(df_prompts))

    assert(len(df.columns) == len(df_prompts.columns) + len(df_prediction.columns) - len(expected_columns))
    print("expected number of columns is correct")

    pd.DataFrame.to_csv(df, os.path.join(path_to_predictions, "full_predictions_with_prompts_" + experiment_id + ".csv"),
                        index=False)

    # extract only articulatory utterances
    df_articulatory = df[df.type == "D"]
    print("\ntotal number of articulatory utterances:", len(df_articulatory))
    pd.DataFrame.to_csv(df_articulatory, os.path.join(path_to_predictions, "df_articulatory.csv"))

    # analysis results:
    print("stops (total, correct, %):", analyse_stops(df_articulatory))
    print("vowels (total, correct, %):", analyse_vowels(df_articulatory))


if __name__ == "__main__":
    main()
