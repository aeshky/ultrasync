"""

Date: Oct 2018
Author: Aciel Eshky
Adapted from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

"""

import os
import numpy as np
import keras


class SyncDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, file_name_list, dir_name, path_to_data, shuffle=True, normalise=False,
                 dim_visual=(5, 63, 412), dim_audio=(1, 20, 12)):
        """ Initialization """

        self.file_name_list = file_name_list
        self.dir_name = dir_name
        self.path_to_data = path_to_data
        self.shuffle = shuffle
        self.normalise = normalise
        self.dim_visual = dim_visual
        self.dim_audio = dim_audio

        self.indexes = np.arange(len(self.file_name_list))
        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """

        return len(self.file_name_list)

    def __getitem__(self, index):
        """ Fetches one batch of data """

        file_name = self.file_name_list[index]

        x, y = self.__data_generation(file_name)

        return x, y

    def on_epoch_end(self):
        """ Updates indexes after each epoch """

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_name):
        """ Generates data containing batch_size samples """

        # print(os.path.join(self.path_to_data, self.dir_name, file_name))

        # I have already places each batch in a file. This function only loads the file.
        samples = np.load(os.path.join(self.path_to_data, self.dir_name, file_name))

        assert samples['ult'].shape[1:] == self.dim_visual and samples['mfcc'].shape[1:] == self.dim_audio
        assert len(samples['ult']) == len(samples['mfcc'] == len(samples['label']))

        # x_name = samples['name']
        x_visual = samples['ult']
        x_audio = samples['mfcc']

        if self.normalise:
            self.min_max_input_norm(x_visual)
            self.min_max_input_norm(x_audio)

        x = [x_visual, x_audio]
        y = [samples['label']]

        return x, y

    @staticmethod
    def min_max_input_norm(x):
        for i, y in enumerate(x):
            for j, v in enumerate(y):
                x[i][j] = (v - v.min()) / (v.max() - v.min())

