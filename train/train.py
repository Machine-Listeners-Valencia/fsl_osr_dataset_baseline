import pickle
from typing import Tuple, Union

import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import tensorflow.keras.backend as K

from utils import utils
from models import baseline_model
from feature_extraction import spectral


def do_training(training_modes: list, feature_extractors: list, train_parameters: dict):
    features_path, l3_path, yamnet_path, mel_spec_path = utils.generate_features_paths()
    evaluation_path, l3_evaluation_path, yamnet_evaluation_path, mel_spec_path_evaluation = utils.generate_evaluation_paths()

    for mode in training_modes:
        if mode == 'full':
            number_of_classes = 24
            for extractor in feature_extractors:
                extractor_data = select_feature_extractor(extractor, mode, l3_path, l3_evaluation_path,
                                                          yamnet_path, yamnet_evaluation_path,
                                                          mel_spec_path, mel_spec_path_evaluation)
                embedding_size, features_root_path, csv_root_storing_path = extractor_data

                execute_trainings(features_root_path, csv_root_storing_path, embedding_size,
                                  number_of_classes, train_parameters)

        elif mode == 'trios':
            number_of_classes = 3
            for extractor in feature_extractors:
                extractor_data = select_feature_extractor(extractor, mode, l3_path, l3_evaluation_path,
                                                          yamnet_path, yamnet_evaluation_path)
                embedding_size, features_root_path, csv_root_storing_path = extractor_data

                for trio_folder in sorted(features_root_path.iterdir()):
                    csv_trio_root_storing_path = csv_root_storing_path / trio_folder.stem
                    if not csv_trio_root_storing_path.is_dir():
                        csv_trio_root_storing_path.mkdir()

                    execute_trainings(trio_folder, csv_trio_root_storing_path, embedding_size,
                                      number_of_classes, train_parameters)


def execute_trainings(features_root_path: Path,
                      csv_root_storing_path: Path,
                      embedding_size: int,
                      number_of_classes: int,
                      train_parameters: dict
                      ) -> None:
    features_folders_to_iter = [features_folder for features_folder in sorted(features_root_path.iterdir())
                                if features_folder.is_dir()]
    for features_folder in features_folders_to_iter:
        print(f"{features_folder.stem}")
        openness = int(features_folder.stem[-1])
        # Training files
        files = [str(item) for item in features_folder.iterdir() if item.is_file()]
        files = utils.natural_sort(files)

        # Check if exist train folder and generate list of folders according to that
        exist_test = check_test_folder(files)
        train_folders = generate_train_folders(files, exist_test)

        # CSV storing path
        csv_path = csv_root_storing_path / f"{features_folder.stem}.csv"
        train_folders, intermediate_training, iterations = check_intermediate_training(csv_path,
                                                                                       train_folders)

        for train_folder in tqdm(train_folders):
            if 'X_val' in locals():
                del X_val
                del y_val

            # Generate splits
            splitted_data = generate_train_val_test_splits(files, train_folder, exist_test)
            try:
                X_train, y_train, X_val, y_val, X_test, y_test = splitted_data
                y_val_test = np.concatenate((y_val, y_test))
            except:
                X_train, y_train, X_val, y_val = splitted_data
                y_test = y_val_test = y_val

            known_labels, unknown_labels, known_indexes, unknown_indexes = split_known_unknown(y_val,
                                                                                               y_val_test)

            # Number of iterations
            number_of_iterations = determine_number_of_iterations(iterations, intermediate_training)
            start_value = iterations[-1] if intermediate_training else 0
            end_value = 5
            K.clear_session()
            for i in tqdm(range(start_value, end_value)):
                # Train model
                if embedding_size is not None:
                    model = baseline_model.BaselineModel(embedding_size, number_of_classes)
                    model.train(X_train,
                                y_train,
                                X_val,
                                y_val,
                                train_parameters['epochs'],
                                train_parameters['batch_size'])
                else:
                    model = baseline_model.OpenSetDCAE(number_of_classes, openness)
                    model.train(X_train,
                                y_train,
                                X_val,
                                y_val,
                                train_parameters['epochs'],
                                train_parameters['batch_size'])

                try:
                    accuracies = calculate_accuracies(exist_test, model, X_val, known_labels, known_indexes,
                                                      unknown_labels, unknown_indexes, X_test, y_test)
                    known_accuracy, unknown_accuracy, test_accuracy = accuracies
                except:
                    accuracies = calculate_accuracies(exist_test, model, X_val, known_labels,
                                                      known_indexes, unknown_labels, unknown_indexes)
                    known_accuracy, unknown_accuracy, test_accuracy = accuracies

                # Save results to csv
                store_to_csv(csv_path, train_folder, i, known_accuracy, unknown_accuracy, test_accuracy)

            # Set intermediate_training to False to continue with next folders
            intermediate_training = False


def select_feature_extractor(extractor: str,
                             mode: str,
                             l3_path: Path,
                             l3_evaluation_path: Path,
                             yamnet_path: Path,
                             yamnet_evaluation_path: Path,
                             mel_spec_path: Path,
                             mel_spec_evaluation_path: Path
                             ) -> Tuple[Union[int, None], Path, Path]:
    if extractor == 'l3':
        embedding_size = 512
        features_root_path = l3_path / mode
        csv_root_storing_path = l3_evaluation_path / mode
    if extractor == 'yamnet':
        embedding_size = 1024
        features_root_path = yamnet_path / mode
        csv_root_storing_path = yamnet_evaluation_path / mode
    if extractor == 'melspectrogram':
        features_root_path = mel_spec_path / mode
        csv_root_storing_path = mel_spec_evaluation_path / mode
        embedding_size = None

    # Create csv_root_storing_path if does not exist
    if not csv_root_storing_path.is_dir():
        csv_root_storing_path.mkdir()

    return embedding_size, features_root_path, csv_root_storing_path


def check_test_folder(files: list) -> bool:
    return True if len(files) % 2 == 1 else False


def generate_train_folders(files: list, exist_test: bool) -> list:
    if exist_test:
        train_folders = [f"feat{i}" for i in range(1, len(files))]
    else:
        train_folders = [f"feat{i}" for i in range(1, len(files) + 1)]

    return train_folders


def check_intermediate_training(csv_path: Path, train_folders) -> Tuple[list, bool]:
    if csv_path.is_file():
        df = pd.read_csv(csv_path, engine='python')
        folder = df['train_fold'].values
        folder = list(np.unique(folder))
        iterations = df['iteration'].values.tolist()
        if iterations[-1] != 5:
            train_folders = utils.natural_sort(utils.lists_diff(folder[:-1], train_folders))
            intermediate_training = True
        else:
            train_folders = utils.natural_sort(utils.lists_diff(folder, train_folders))
            intermediate_training = False
    else:
        intermediate_training = False
        iterations = [0]

    return train_folders, intermediate_training, iterations


def generate_train_val_test_splits(files: list,
                                   train_folder: str,
                                   exist_test: bool
                                   ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray, np.ndarray],
                                              Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    for i, filepath in enumerate(files):
        data = pickle.load(open(filepath, 'rb'))
        vars()[f"feat{i + 1}"] = data['features']
        vars()[f"target{i + 1}"] = data['labels']

        if f"feat{i + 1}" == train_folder:
            X_train = vars()[f"feat{i + 1}"]
            y_train = vars()[f"target{i + 1}"]
        else:
            if 'X_val' not in locals():
                X_val = vars()[f"feat{i + 1}"]
                y_val = vars()[f"target{i + 1}"]
            else:
                if not exist_test:
                    X_val = np.concatenate((X_val, vars()[f"feat{i + 1}"]))
                    y_val = np.concatenate((y_val, vars()[f"target{i + 1}"]))
                else:
                    if len(files) == i + 1:
                        X_test = vars()[f"feat{i + 1}"]
                        y_test = np.zeros((X_test.shape[0], vars()[f"target1"].shape[1]))
                    else:
                        X_val = np.concatenate((X_val, vars()[f"feat{i + 1}"]))
                        y_val = np.concatenate((y_val, vars()[f"target{i + 1}"]))

    try:
        return X_train, y_train, X_val, y_val, X_test, y_test
    except:
        return X_train, y_train, X_val, y_val


def split_known_unknown(y_val: np.ndarray, y_val_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list, list]:
    # Geet known/unknown indexes
    known_indexes = [i for i, item in enumerate(y_val) if np.sum(item) != 0]
    unknown_indexes = [i for i, item in enumerate(y_val_test) if np.sum(item) == 0]

    # Split labels into known/unknown
    known_labels = y_val_test[known_indexes, :]
    unknown_labels = y_val_test[unknown_indexes, :]

    return known_labels, unknown_labels, known_indexes, unknown_indexes


def determine_number_of_iterations(iterations: list, intermediate_training: bool) -> int:
    return 5 - iterations[-1] if intermediate_training else 5


def calculate_accuracies(exist_test: bool,
                         model: baseline_model.BaselineModel,
                         X_val: np.ndarray,
                         known_labels: np.ndarray,
                         known_indexes: list,
                         unknown_labels: np.ndarray,
                         unknown_indexes: list,
                         X_test: Union[np.ndarray, None] = None,
                         y_test: Union[np.ndarray, None] = None
                         ) -> Tuple[float, float, float]:
    # Make predictions and evaluate
    if exist_test:
        predictions_openset_val = model.openset_predict((np.concatenate((X_val, X_test))))
        predictions_openset_test = model.openset_predict(X_test)
        # Unknown unknown accuracy
        test_accuracy = accuracy_score(y_test, predictions_openset_test)
    else:
        predictions_openset_val = model.openset_predict(X_val)

    # Split validation predictions into known and unknown ground truth indexes
    predictions_known_labels = predictions_openset_val[known_indexes, :]
    predictions_unknown_labels = predictions_openset_val[unknown_indexes, :]

    # Known known and known unknown accuracy
    known_accuracy = accuracy_score(known_labels, predictions_known_labels)
    unknown_accuracy = accuracy_score(unknown_labels, predictions_unknown_labels)

    # Assign known unknown to unknown unknown if X_test does not exist
    if not exist_test:
        test_accuracy = unknown_accuracy

    return known_accuracy, unknown_accuracy, test_accuracy


def store_to_csv(path: Path,
                 train_folder: str,
                 i: int,
                 known_accuracy: float,
                 unknown_accuracy: float,
                 test_accuracy: float) -> None:
    folder = [train_folder]
    iteration = [i + 1]
    accuracies_known = [known_accuracy]
    accuracies_unknown = [unknown_accuracy]
    accuracies_test = [test_accuracy]

    metadata = {'train_fold': folder, 'iteration': iteration, 'known_acc': accuracies_known,
                'unknown_acc': accuracies_unknown, 'test_acc': accuracies_test}
    metadata_df = pd.DataFrame(metadata, columns=['train_fold', 'iteration', 'known_acc', 'unknown_acc', 'test_acc'])

    if path.is_file():
        metadata_df.to_csv(path_or_buf=path, mode='a', index=False, header=False)
    else:
        metadata_df.to_csv(path_or_buf=path, mode='a', index=False, header=True)
