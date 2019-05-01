"""

Date: Oct 2018
Author: Aciel Eshky

A script to predict the offset for utterances given a set of "offset candidates".
The script uses the model to predict the distance for each offset candidate,
then select the candidate with the smallest average distance.

Given a set of offset candidates
for each utterance:
    for each offset:
        align the two signals according to the given candidate
        produce the non-overlapping windows of ultrasound and MFCC pairs
        use the model to predict the Euclidean distance for each pair
        average the distances across utterance
    select the offset with the smallest average distance as the prediction

"""

import sys
import os

import json
import numpy as np
import pandas as pd

from ustools.core import UltraSuiteCore
from ustools.chunk import Chunk
from synchronisation.experiment import SyncExperiment


def compute_aggregate_distance(dir, utterance_file_basename, model, offset):
    """
    Computes the aggregate distance over chunks of an utterance given an offset.
    Can be invoked using "mean" distance or "sum" of distances.
    Warning: Can return None. This can happen when the offset causes one of the modalities to become of zero length.
    :param dir:
    :param utterance_file_basename:
    :param model:
    :param offset:
    :param aggregation_method:
    :return:
    """
    try:

        core = UltraSuiteCore(dir, utterance_file_basename)
        core.params["sync"] = offset
        core.process(apply_sync_param=True, ult_frame_skip=True, stride=5,
                     transform_ult=False, apply_vad=False, remove_zero_regions=True, down_sample_ult=True)

        chunk = Chunk(core, ult_chunk_size=5, mfcc_feat=True, drop_first_mfcc=False,
                      fbank_feat=False, transform_ult=False)

        distances = model.predict_on_batch([chunk.ult_chunks, chunk.mfcc_chunks])

        if len(distances) > 0:
            average_distance = np.mean(distances)
            # median_distance = np.median(distances)
            # min_distance = np.min(distances)
            # confidence = median_distance - min_distance
            # print(len(distances), "> 0:", average_distance, median_distance, min_distance, confidence)
        else:
            average_distance = None
            # confidence = 0
            # print(len(distances), "<= 0:", average_distance, confidence)

        # return [average_distance, confidence]
        return average_distance

    except (ValueError, AttributeError) as e:

        print("Error:", e, "returning [None, 0]")
        # return [None, 0]
        return None


def select_min_distance_offset(path_to_data, file_basename, model, offset_candidates):

    print("processing:", os.path.join(path_to_data, file_basename))

    smallest_average_distance = sys.maxsize
    # confidence_of_smallest_average_distance = 0
    offset_prediction = None

    for offset in offset_candidates:

        # [average_distance, confidence] = compute_aggregate_distance(path_to_data, file_basename, model, offset)
        average_distance = compute_aggregate_distance(path_to_data, file_basename, model, offset)

        if average_distance and average_distance < smallest_average_distance:
            smallest_average_distance = average_distance
            offset_prediction = offset
            # confidence_of_smallest_average_distance = confidence

    # return offset_prediction, smallest_average_distance, confidence_of_smallest_average_distance
    return offset_prediction, smallest_average_distance


def get_difference(value_a, value_b):
    try:
        return value_a - value_b
    except TypeError:
        return None


def is_correct(value, threshold=0.045):
    """
    If the difference is less than 45 milliseconds (0.045 seconds) then it's correct. Otherwise, it's incorrect.
    :param value:
    :param threshold:
    :return:
    """
    try:
        if abs(value) <= threshold:
            return 1
        else:
            return 0
    except TypeError:
        return None


def main():

    # old 20190123-14hr31m59s 0.6470
    # latest 20190207-09hr30m13s 0.6555

    # input
    # experiment id
    # input folder
    # output folder
    # validation utterances data frame

    arguments_file = sys.argv[1]  # "config_FF.json" or "configCNN.json"

    with open(arguments_file, "r") as read_file:
        args = json.load(read_file)

    exp = SyncExperiment(args["experiment_id"])
    exp.load_model(args["model_dir"])
    print(exp, "model loaded.")

    # offset candidates come from the training data with a bin size of 0.045 in the range of 0 - 1.8
    # empty bins were discarded for efficiency

    offset_candidates = [0.0, 0.18, 0.225, 0.27, 0.315, 0.36, 0.405, 0.45, 0.495, 0.54, 0.585, 0.72, 0.765, 0.81, 0.945,
                         0.99, 1.215, 1.26, 1.305, 1.35, 1.395, 1.665, 1.71, 1.755]
    # 24 options

    for params_df, subset_name in zip(args["params_df"], args["subset_name"]):

        out_file = os.path.join(args["output_dir"],
                                "offset_predictions_" + subset_name + "_" + exp.experiment_id +
                                "_" + str(args["bin_size"]) + ".csv")

        if os.path.isfile(out_file):

            print(out_file, "already existing. Skipping to next subset.")

        else:

            print("loading dataframe:", params_df)
            df = pd.read_csv(params_df)

            print("computing offset.")

            results = []
            for index, row in df.iterrows():
                result = list(select_min_distance_offset(row['path'], row['base_file_name'],
                                                         exp.model, offset_candidates))

                results.append([row['path']] +
                               [row['base_file_name']] +
                               [row['type']] +
                               [row['dataset']] +
                               [row['speaker']] +
                               [row['session']] +
                               [row['offset_true']] +
                               [row['offset_true_bin']] +
                               result)

            cols = ['path', 'base_file_name', 'type', 'dataset', 'speaker', 'session',
                    'offset_true', 'offset_true_bin',
                    'offset_prediction',
                    'smallest_average_distance']
            df_results = pd.DataFrame(results, columns=cols)

            pd.DataFrame.to_csv(df_results, out_file, index=False)
            print("df saved.")

            # additional computations...
            df_results['difference'] = df_results.apply(lambda row: row['offset_prediction'] - row['offset_true'],
                                                        axis=1)

            # df_results['correct'] = df_results.apply(lambda row: is_correct(row['difference']), axis=1)

            df["detectability_correct"] = df.difference.apply(lambda x: 1 if (x < 0.045) and (x > -0.125) else 0)

            pd.DataFrame.to_csv(df_results, out_file, index=False)
            print("evaluation 1 saved.")

            df_results['same_bin'] = df_results.apply(lambda row:
                                                      1 if row['offset_true_bin'] == row["offset_prediction"] else 0,
                                                      axis=1)

            pd.DataFrame.to_csv(df_results, out_file, index=False)
            print("evaluation 2 saved.")


if __name__ == "__main__":
    main()
