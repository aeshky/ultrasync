"""

Date: Jul 2018
Author: Aciel Eshky

These functions are invoked from the script create_sync_.

In function create_samples, core.process prepares the data.

Data preparation steps:
1) Excluding non-spoken utterances. These are utterances ``Non-speech" (E) which include coughs and swallows.
2) Applying true synchronisation offset. This crops the leading audio and trimming the end of the longer modality.
3) Reducing frame rate. From 121.5 fps to 24.3 fps by retaining 1 out of every 5 frames.
4) Down-sampling frames. by a factor of (1, 3), shrinking the frame size from 63x412 to 63x138 using max pixel value.
5) Removing zero regions from audio and corresponding ultrasound. Zero regions come from anonymisation.

Not applied: 6) Removing regions of silence using VAD. This step reduced performance considerably and is therefore
skipped.

"""

import os
import numpy as np
import copy
import itertools
import pandas as pd

from ustools.core import UltraSuiteCore
from ustools.chunk import Chunk


def apply_and_concat(dataframe, field, func, column_names):
    """
    Not currently used.
    Taken from: https://stackoverflow.com/a/44292630/5190279
    :param dataframe:
    :param field:
    :param func: the function to be applied to the 'field' column
    :param column_names: the return column names
    :return: retuns a new dataframe with old field and new fields concatenated.
    """
    return pd.concat((
        dataframe,
        dataframe[field].apply(
            lambda cell: pd.Series(func(cell), index=column_names))), axis=1)


def interleave_lists(a, b):
    """
    Interleaves the values of two lists. The length is equal to the shorter of the two lists x 2

    :param a: first list e.g., ['a', 'b', 'c', 'd', 'e']
    :param b: second list e.g., [1, 2, 3]
    :return: interleaved list ['a', 1, 'b', 2, 'c', 3]
    """
    return list(itertools.chain(*zip(a, b)))


def create_samples(path_to_utterance, utterance_basename):
    """
    A function to create positive and negative samples from utterances.

    The samples correspond to 200 ms of ultrasound and audio.

    Positive samples are pairs of ultrasound and corresponding MFCC frames.
    To create negative samples, the function randomises pairings of ultrasound to MFCC frames.
    The function generates as many negative as positive samples to achieve a balanced dataset.

    :param path_to_utterance: the directory where the utterance tuple is stored
    :param utterance_basename: the basename of the utterance tuple
    :return: a list of a zipped structure. Each zipped structure corresponds to one sample and contains:
            1) chunk_id,
            2) label (positive or negative),
            3) 2D mfcc matrix (1, 20, 12),
            4) and 3D ult matrix (5, 63, 412)
    """

    # create core object
    core = UltraSuiteCore(path_to_utterance, utterance_basename)

    # data preparation
    core.process(apply_sync_param=True, ult_frame_skip=True, stride=5,
                 transform_ult=False, apply_vad=False, remove_zero_regions=True,
                 down_sample_ult=True)

    # create chunk object, only mfcc and original ultrasound (this can potentially change)
    chunk = Chunk(core, ult_chunk_size=5, mfcc_feat=True, drop_first_mfcc=False, fbank_feat=False, transform_ult=False)

    chunk_ids = chunk.chunk_ids
    mfcc = chunk.mfcc_chunks
    ult = chunk.ult_chunks

    # make a copy of the mfcc
    mfcc_shuffled = copy.copy(mfcc)

    # shuffle the copy
    np.random.shuffle(mfcc_shuffled)

    # interleave positive and negative examples
    chunk_ids = np.array(interleave_lists(chunk_ids, chunk_ids))
    all_ult = np.array(interleave_lists(ult, ult))
    all_mfcc = np.array(interleave_lists(mfcc, mfcc_shuffled))

    # give label (positive = 1.0 or negative = 0.0)
    label = np.array(interleave_lists(np.ones(shape=mfcc.shape[0]), np.zeros(shape=mfcc_shuffled.shape[0])))

    # add a label (positive or negative) to the chunk id
    chunk_ids = np.array([id + "_" + ("pos" if bool(j) else "neg") for id, j in zip(chunk_ids, label)])

    assert len(all_ult) == len(all_mfcc) == len(label) == len(chunk_ids)

    # return zipped structure
    return list(zip(chunk_ids, label, all_ult, all_mfcc))


def save_samples_to_disk(samples, root_id, output_dir):
    """
    This function saves the samples created by create_samples to specified output_dir on disk.

    :param samples: samples created using create_samples
    :param root_id: the root id: dataset_speaker_session_filename
    :param output_dir: the output directory
    :return: a list of names of saved samples.
    """

    # store the names of the created samples
    chunk_names = []

    # iterate over the samples and save each as a separate file.
    for chunk_id, label, ult, mfcc in samples:

        sample_id = root_id + "-" + chunk_id

        np.savez(os.path.join(output_dir, sample_id), label=label, ult=ult, mfcc=mfcc)

        chunk_names.append(sample_id)

    return chunk_names

