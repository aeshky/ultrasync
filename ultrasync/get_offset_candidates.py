"""

Date: Nov 2018
Author: Aciel Eshky

A script that retrieves the true offsets in the training set and places them in bins to get offset candidates
for prediction.

"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def main():

    # We chose the bin size to be the smaller of the absolute values of the detectability boundaries -125 and 45 ms
    bin_size = 0.045

    train_1 = pd.read_csv("~/Documents/synchronisation/sync_10/offset_true/df_offset_train_1.csv")
    train_2 = pd.read_csv("~/Documents/synchronisation/sync_10/offset_true/df_offset_train_2.csv")
    train_3 = pd.read_csv("~/Documents/synchronisation/sync_10/offset_true/df_offset_train_3.csv")

    df = pd.concat([train_1, train_2, train_3])
    df = df.reset_index()

    print(df.shape)
    min_val = min(df["offset_true"])
    max_val = max(df["offset_true"])

    print("min val:", min_val, "- max val:", max_val)

    candidates = np.round(np.arange(min_val, max_val + bin_size, bin_size), 3)

    print(len(candidates))
    print(candidates)

    x = plt.hist(df["offset_true"], bins=candidates)
    plt.show()

    candidates_val = []
    for i, j in zip(x[0], x[1]):
        print(i, j)
        if i > 0:
            candidates_val.append(j)

    print(len(candidates_val))
    print(candidates_val)

    print("probabilities:")
    y = np.add(x[0], 1)
    s = y.sum()
    for i, j in zip(x[1], y):
        print(i, j/s)

    test_1 = pd.read_csv("~/Documents/synchronisation/sync_10/offset_true/df_offset_test_1.csv")
    test_2 = pd.read_csv("~/Documents/synchronisation/sync_10/offset_true/df_offset_test_2.csv")
    test_3 = pd.read_csv("~/Documents/synchronisation/sync_10/offset_true/df_offset_test_3.csv")
    test_4 = pd.read_csv("~/Documents/synchronisation/sync_10/offset_true/df_offset_test_4.csv")
    test_5 = pd.read_csv("~/Documents/synchronisation/sync_10/offset_true/df_offset_test_5.csv")

    df = pd.concat([test_1, test_2, test_3, test_4, test_5])
    df = df.reset_index()

    # min_val = min(df["offset_true"])
    # max_val = max(df["offset_true"])

    print("min val:", min_val, "- max val:", max_val)

    candidates = np.round(np.arange(min_val, max_val + bin_size, bin_size), 3)

    print(len(candidates))
    print(candidates)

    x = plt.hist(df["offset_true"], bins=candidates)
    plt.show()


if __name__ == "__main__":
    main()
