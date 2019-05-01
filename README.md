# UltraSync Documentation 

This README file describes the order in which the scripts are run to reproduce the results of the 2019 Interspeech paper: _"Synchronising audio and ultrasound by learning cross-modal embeddings"_

## Data preparation and self-supervision sample creation:

1. **create_sync_samples.py** creates positive and negative samples from the original UltraSuite data. Samples are 200 ms of ultrasound and corresponding MFCC.

2. **get_sync_samples_info.py** counts the number of samples for each data set, speaker, and session allowing us to choose training, validation, and testing splits based on the number of samples. No information other than the sizes of subset is used to split the data. 

3. **create_train_test_splits.py** creates data frames with the names of samples in each subset (train, val, or test). It also shuffles the order of samples in the training set to prepare for the next step.

## Model Training:

4. **create_experiment_data.py** uses the data frames from the previous step to create batches of samples. We experimented with a few batch sizes and eventually used 64 (32 positive and 32 negative for the same samples).

5. **run_model_gpu.py** trains the model. It outputs a model file and a results file. The results file reports the loss on training, validation and test data. It also reportes a simple binary classification accuracy by placing a threshold of 0.5 on the predicted distances. 

## Prediction and evaluation:

6. **get_true_offset.py** retrieves the true offsets for training, validation and test sets. The true offsets in the test set serves as ground truth values for evaluation.

7. **get_offset_candidates.py** retrieves the true offsets from the training data and bins them to get "offset candidates" for prediction.

8. **predict_offset.py** uses the model to predict the distance for each candidate and then selects the candidate with the smallest average distance. It also calculates the distance between truth and prediction to get the overall accuracy.

9. **analyse_offset_prediction.py** analyses the results across different attributes calculating the mean discrepancy in addition to accuracy

10. **random_prediction.py** gets the accuracy for a random prediction which serves as a lower-bound for evaluation.
