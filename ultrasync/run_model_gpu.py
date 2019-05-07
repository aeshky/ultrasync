"""

Date: Oct 2018
Author: Aciel Eshky
Adapted from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

"""

# GPU set up:
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# or type this in the terminal:
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=3


import random
random.seed(42)
from numpy.random import seed as np_seed
np_seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

import os
import sys
from datetime import datetime

import json
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from ultrasync.sync_data_generator import SyncDataGenerator
from ultrasync.two_stream_model import TwoStreamNetwork, accuracy, contrastive_loss
from ultrasync.experiment import SyncExperiment


def compute_metrics(generator, model, steps):

    print("computing distances..")

    pos = []
    neg = []

    for i, (x, y) in enumerate(generator):
        if i == steps:
            break
        predictions = model.predict_on_batch(x)

        pos.extend(predictions[y[0] == 1.0][:, 0])
        neg.extend(predictions[y[0] == 0.0][:, 0])
    print(i, "batches processed.")

    result = dict()
    result["pos_distances"] = [float(n) for n in pos]
    result["neg_distances"] = [float(n) for n in neg]

    print("computing additional metrics..")
    additional_metrics = model.evaluate_generator(generator, steps=steps)
    result.update({m: float(val) for m, val in zip(model.metrics_names, additional_metrics)})

    print("Metrics computed.")

    return result


def main():

    config_filename = sys.argv[1]  # "config_FF.json" or "configCNN.json"
    with open(config_filename, "r") as read_file:
        config = json.load(read_file)

    config["dim_visual"] = tuple(config["dim_visual"])
    config["dim_audio"] = tuple(config["dim_audio"])

    # get a unique experiment id using date and time
    experiment_id = datetime.now().strftime("%Y%m%d-%Hhr%Mm%Ss")
    config["experiment_id"] = experiment_id

    # print the configuration
    print(config)

    # take only a subset of the train and val data
    data_subset = config["data_subset"]

    files_train = os.listdir(os.path.join(config['path_to_data'], config['train_folder']))
    files_train.sort()
    if data_subset:
        files_train = files_train[:data_subset]

    files_val = os.listdir(os.path.join(config['path_to_data'], config['val_folder']))
    files_val.sort()
    if data_subset:
        files_val = files_val[:(data_subset * 2):2]

    files_test = os.listdir(os.path.join(config['path_to_data'], config['test_folder']))
    files_test.sort()
    if data_subset:
        files_test = files_test[:(data_subset * 2):2]

    partition = {'train': [f for f in files_train if f.startswith('train') and f.endswith('.npz')],
                 'val': [f for f in files_val if f.startswith('val') and f.endswith('.npz')],
                 'test': [f for f in files_test if f.startswith('test') and f.endswith('.npz')]}

    params = {'path_to_data': config['path_to_data'],
              'shuffle': config['shuffle'],
              'normalise': config['min_max_normalisation'],
              'dim_visual': config['dim_visual'],
              'dim_audio': config['dim_audio']}

    # Data generators
    training_generator = SyncDataGenerator(partition['train'], config['train_folder'], **params)
    validation_generator = SyncDataGenerator(partition['val'], config['val_folder'], **params)
    test_generator = SyncDataGenerator(partition['test'], config['test_folder'], **params)

    # initialise model
    print("Initialising model.")
    network = TwoStreamNetwork(config)
    model = network.model

    if config['load_existing_model']:
        exp_existing = SyncExperiment(config['load_existing_model'])
        exp_existing.load_model(config['existing_model_location'])
        model = exp_existing.model

    exp = SyncExperiment(experiment_id)

    learning_rate = config["learning_rate"]  # 1, 0.1, 0.01, 0.001

    assert config['optimiser'] in ["SGD", "Adam"]
    if config['optimiser'] == "Adam":
        opt = Adam(lr=learning_rate)
    else:
        opt = SGD(lr=learning_rate)

    if config['loss_function'] == "contrastive_loss":
        model.compile(loss=contrastive_loss, optimizer=opt, metrics=[accuracy])
    else:
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=[accuracy])

    epochs = config["num_epochs"]

    callbacks = []
    if config["reduce_lr"]:
        reduce_lr = ReduceLROnPlateau(monitor=config["monitor"], factor=0.1, patience=2, verbose=1, mode='auto',
                                      min_delta=0.0001, cooldown=0, min_lr=0)
        callbacks.append(reduce_lr)

    if config["model_checkpoint"]:
        checkpoint_file = os.path.join(config["path_to_output"], exp.best_model_weights_filename)
        model_checkpoint = ModelCheckpoint(checkpoint_file,
                                           monitor='val_loss', verbose=0, save_best_only=True,
                                           save_weights_only=True, mode='auto', period=1)
        callbacks.append(model_checkpoint)

    # Train model on dataset
    print("Training.")
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=len(training_generator),
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=len(validation_generator),
                        use_multiprocessing=False,
                        verbose=2,
                        callbacks=callbacks)  # [reduce_lr, model_checkpoint])

    if config["save_config"]:
        exp.config = config
        exp.save_config(config['path_to_output'])
        print("Config file saved to disk.")

    if config["save_model"]:
        exp.model = model
        exp.save_model(config['path_to_output'])
        print("Model saved to disk.")

    if config["save_results"]:
        metrics = dict()
        metrics["config"] = config
        print("Computing metrics on training data.")
        metrics["train"] = compute_metrics(training_generator, model, steps=len(training_generator))
        print("Computing metrics on validation data.")
        metrics["val"] = compute_metrics(validation_generator, model, steps=len(validation_generator))
        print("Computing metrics on test data.")
        metrics["test"] = compute_metrics(test_generator, model, steps=len(test_generator))

        exp.results = metrics
        exp.save_results(config['path_to_output'])
        print("Results saved to disk.")


if __name__ == "__main__":
    main()



