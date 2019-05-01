"""

Date: Nov 2018
Author: Aciel Eshky

A class which saves the experiment details.

"""
import os
import json
from keras.models import model_from_json


class SyncExperiment:

    def __init__(self, experiment_id):

        self.experiment_id = experiment_id

        self.config_filename = "config_" + experiment_id + ".json"
        self.model_architecture_filename = "model_" + experiment_id + ".json"
        self.model_weights_filename = "model_" + experiment_id + ".h5"
        self.best_model_weights_filename = "best_model_" + experiment_id + ".h5"
        self.results_filename = "results_" + experiment_id + ".json"

        self.config = None
        self.model = None
        self.architecture = None
        self.results = None

    def load_config(self, path):
        """

        :param path:
        :return:
        """
        with open(os.path.join(path, self.config_filename), "r") as in_file:
            self.config = json.load(in_file)
            in_file.close()

    def save_config(self, path):
        """

        :param path:
        :param config:
        :return:
        """
        with open(os.path.join(path, self.config_filename), "w") as out_file:
            json.dump(self.config, out_file, indent=4)
            out_file.close()

    def save_model(self, path):
        """

        :param path:
        :param model:
        :return:
        """
        with open(os.path.join(path, self.model_architecture_filename), "w") as out_file:
            self.architecture = self.model.to_json()
            out_file.write(self.architecture)
            out_file.close()

        self.model.save_weights(os.path.join(path, self.model_weights_filename))

    def load_model(self, path):
        """

        :param path:
        :return:
        """
        with open(os.path.join(path, self.model_architecture_filename), "r") as in_file:
            self.architecture = in_file.read()
            in_file.close()

            self.model = model_from_json(self.architecture)
            self.model.load_weights(os.path.join(path, self.model_weights_filename))

    def save_results(self, path):
        """

        :param path:
        :return:
        """
        with open(os.path.join(path, self.results_filename), 'w') as out_file:
            json.dump(self.results, out_file)
            out_file.close()

    def load_results(self, path):
        """

        :param path:
        :return:
        """
        with open(os.path.join(path, self.results_filename)) as in_file:
            self.results = json.load(in_file)
            in_file.close()

