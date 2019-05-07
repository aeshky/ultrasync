"""

Date: Mar 2019
Author: Aciel Eshky

Random prediction gives us a lower bound for performance.

"""

import os
import sys
import pandas as pd
import numpy as np


def main():

    path = sys.argv[1]  # path to where true offsets csv files are stored "../offset_true"

    candidates = ['0.0', '0.18', '0.225', '0.27', '0.315',
                  '0.36', '0.405', '0.45', '0.495', '0.54', '0.585', '0.72', '0.765',
                  '0.81', '0.945', '0.99', '1.215', '1.26', '1.305', '1.35', '1.395',
                  '1.665', '1.71', '1.755']

    print("number of candidates:", len(candidates))

    files = [i for i in os.listdir(path) if i.startswith("df_offset_test_")]

    df = pd.DataFrame()
    for f in files:
        df = pd.concat([df, pd.read_csv(os.path.join(path, f))])

    print("number of utterances in data frame:", df.shape[0])

    candidates_double = np.array(candidates).astype(float)

    accuracy = []
    difference_mean = []
    difference_std = []

    n = 1000
    print("predicting randomly", n, "times")
    for i in range(0, n):

        df['random'] = np.random.choice(candidates_double, 1502)
        df['difference'] = df.apply(lambda row: (row['random'] - row['offset_true']) * 1000, axis=1)
        df["random_correct"] = df.difference.apply(lambda d: 1 if -125 < d < 45 else 0)

        res = df.random_correct.sum() / df.shape[0]
        accuracy.append(res)

        x = df.difference.describe()
        difference_mean.append(x['mean'])
        difference_std.append(x['std'])

    print("difference summary (only last iteration):", df.difference.describe())

    print("mean accuracy:", round(np.mean(accuracy) * 100, 1), "%")

    print("mean mean difference:", round(np.mean(difference_mean)))

    print("mean std difference:", round(np.mean(difference_std)))


if __name__ == "__main__":
    main()
