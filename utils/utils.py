from typing import Tuple, Union
import random
import re

from pathlib import Path
import pandas as pd
import numpy as np
from natsort import  natsorted


def prepare_paths():
    cwd_path = Path.cwd()

    # Data paths
    data_path = cwd_path / 'data'
    metadata_path = data_path / 'meta'
    pattern_path = data_path / 'pattern_sounds'
    unwanted_path = data_path / 'unwanted_sounds'

    return data_path, metadata_path, pattern_path, unwanted_path


def generate_features_paths(root_path: Path = Path.cwd()) -> Tuple[Path, Path, Path]:
    features_path = root_path / 'data/features'
    l3_path = features_path / 'l3_features'
    yamnet_path = features_path / 'yamnet_features'

    if not features_path.is_dir():
        features_path.mkdir()
    if not l3_path.is_dir():
        l3_path.mkdir()
        generate_full_trios_path(l3_path)

    if not yamnet_path.is_dir():
        yamnet_path.mkdir()

    return features_path, l3_path, yamnet_path


def generate_evaluation_paths(root_path: Path = Path.cwd()) -> Tuple[Path, Path, Path]:
    evaluation_path = root_path / 'data/evaluation'
    l3_path = evaluation_path / 'l3_evaluation'
    yamnet_path = evaluation_path / 'yamnet_evaluation'

    if not evaluation_path.is_dir():
        evaluation_path.mkdir()
    if not l3_path.is_dir():
        l3_path.mkdir()
        generate_full_trios_path(l3_path)

    if not yamnet_path.is_dir():
        yamnet_path.mkdir()

    return evaluation_path, l3_path, yamnet_path


def generate_full_trios_path(path: Path) -> None:
    path_full = path / 'full'
    path_trios = path / 'trios'
    path_full.mkdir()
    path_trios.mkdir()


def recover_csv_meta_path(root_path: Path = Path.cwd()) -> Path:
    return root_path / 'data/meta/csv'


def folders_in_path(path: Path) -> list:
    return [folder for folder in sorted(path.iterdir()) if folder.is_dir()]


def prepare_extraction_options(pattern_path: Path) -> Tuple[list, dict, list, int, int]:
    # Labels
    pattern_folders = pattern_path.glob('**/')
    labels = [f"pattern_{str(i).zfill(2)}" for i, _ in enumerate(pattern_folders)]
    labels = ["unknown"] + labels

    # Few shot options
    shots = {1: 40, 2: 20, 4: 10}

    # Open-set options
    openness = [0, 50, 100]

    # Other configuration options
    number_of_classes = 25
    number_of_folder_files = 40  # number of files per each origin folder

    return labels, shots, openness, number_of_classes, number_of_folder_files


def prepare_files_and_labels(path: Path, tags: list, seed: int = 42) -> Tuple[list, list]:
    cwd_path = Path.cwd()

    files = []
    labels = []
    folders = [folder for folder in sorted(path.iterdir()) if folder.is_dir()]
    for folder, tag in zip(folders, tags[1:]):
        audios = list(folder.glob('*.wav'))
        audios = [str(item.relative_to(cwd_path)) for item in audios]
        audios.sort()
        random.seed(seed)
        audios = random.sample(audios, k=len(audios))

        label = [tag] * len(audios)

        files.extend(audios)
        labels.extend(label)

    return files, labels


def prepare_pattern_trios_indexes(number_of_folder_files: int, number_of_classes: int, seed: int = 42) -> list:
    np.random.seed(seed=seed)
    number_of_positive_classes = number_of_classes - 1  # take into account only pattern classes
    random_group = np.random.permutation(number_of_positive_classes)
    random_group = list(random_group)

    triplet_index = []
    for index in range(number_of_positive_classes):
        match_index = find(random_group, index)
        match_index = match_index[0]
        group = match_index // 3
        triplet_index.append(group)

    trios_indexes = list(np.repeat(triplet_index, number_of_folder_files))

    return trios_indexes


def prepare_unknown_trios_indexes(unknown_files: list) -> list:
    return [8] * len(unknown_files)


def prepare_train_folds(shots: dict, number_of_folder_files: int, folders: list) -> dict:
    folds = {}
    for shot, number_of_folds in shots.items():
        number_of_train_files = round(
            number_of_folder_files / number_of_folds)  # number of each class files in each train folder

        fold = []
        for i in range(number_of_folds):
            fold.extend([i + 1] * number_of_train_files)

        folds[shot] = fold * len(folders)

    return folds


def prepare_unknown_files_and_labels(path: Path, seed: int = 42) -> Tuple[list, list]:
    cwd_path = Path.cwd()

    files = []
    labels = []
    folders = [folder for folder in sorted(path.iterdir()) if folder.is_dir()]
    for folder in folders:
        audios = list(folder.glob('*.wav'))
        audios = [str(item.relative_to(cwd_path)) for item in audios]
        audios.sort()
        random.seed(seed)
        audios = random.sample(audios, k=len(audios))

        label = ['unknown'] * len(audios)

        files.extend(audios)
        labels.extend(label)

    return files, labels


def match_substring(elements_list: list, match_pattern: str) -> list:
    """
    Return indexes of a list that matches with pattern. Similar to MATLAB find function
    """
    indexes = [i for i in range(len(elements_list)) if match_pattern in elements_list[i]]

    return indexes


def find(elements_list: list, match_pattern: str) -> list:
    """
    Return indexes of a list that matches with pattern. Similar to MATLAB find function
    """
    indexes = [i for i in range(len(elements_list)) if elements_list[i] == match_pattern]

    return indexes


def create_configuration_csvs(pattern_files: list,
                              unknown_files: list,
                              pattern_labels: list,
                              unknown_labels: list,
                              pattern_train_folds: list,
                              unknown_train_folds: list,
                              pattern_trios_indexes: list,
                              unknown_trios_indexes: list,
                              shots: dict,
                              openness_factors: list,
                              metadata_path: Path
                              ) -> None:
    files = pattern_files + unknown_files
    labels = pattern_labels + unknown_labels
    trios_indexes = pattern_trios_indexes + unknown_trios_indexes
    for shot, number_of_folds in shots.items():
        train_folds = pattern_train_folds[shot] + unknown_train_folds[shot]
        for openness in openness_factors:
            if openness == 0:
                test_classes_list = []
            elif openness == 50:
                test_classes_list = ['clapping', 'door_slam', 'water', 'music', 'pots_and_pans']
                number_of_test_folds = number_of_folds + 1
            elif openness == 100:
                test_classes_list = ['car_horn', 'clapping', 'cough', 'door_slam', 'engine',
                                     'keyboard_tap', 'music', 'pots_and_pans', 'steps', 'water']
                number_of_test_folds = number_of_folds + 1

            for class_name in test_classes_list:
                indexes = match_substring(files, class_name)

                for i in indexes:
                    train_folds[i] = number_of_test_folds

            csv_root_path = metadata_path / 'csv'
            if not csv_root_path.is_dir():
                csv_root_path.mkdir()
            csv_path = csv_root_path / f"few_shot_k_{shot}_openness_{openness}.csv"
            metadata = {'filename': files, 'target': labels, 'fold': train_folds, 'trio_index': trios_indexes}
            metadata_df = pd.DataFrame(metadata, columns=['filename', 'target', 'fold', 'trio_index'])
            metadata_df.to_csv(path_or_buf=csv_path, index=False, header=True)


def convert_to_one_hot(labels: list, mode) -> np.ndarray:
    label_ids = np.array(convert_labels_to_ids(labels))

    if mode == 'full':
        # Consider 25 classes if 0 class (unknown) is present
        number_of_labels = len(unique(label_ids))
    elif mode == 'trios':
        label_ids, number_of_labels = map_to_trios_labels(label_ids)

    # Convert to one hot
    one_hot_array = np.zeros((label_ids.shape[0], number_of_labels))

    unique_labels = unique(label_ids)
    if 0 in unique_labels:
        # Case pattern + unknown files
        one_hot_array[np.arange(label_ids.size), label_ids] = 1
        return one_hot_array[:, 1:]
    else:
        # Case only pattern files
        one_hot_array[np.arange(label_ids.size), label_ids-1] = 1
        return one_hot_array


def map_to_trios_labels(label_ids: list) -> Tuple[np.ndarray, int]:
    unique_labels = unique(label_ids)
    if 0 in unique_labels:
        labels_mapper = {label: i for i, label in enumerate(unique_labels)}
        label_ids = [labels_mapper[label] for label in label_ids]
    else:
        labels_mapper = {label: i + 1 for i, label in enumerate(unique_labels)}
        label_ids = [labels_mapper[label] for label in label_ids]

    number_of_labels = len(unique_labels)

    return np.array(label_ids), number_of_labels


def unique(elements: list) -> list:
    unique_elements = list(set(elements))
    unique_elements.sort()

    return unique_elements

def convert_labels_to_ids(labels: list) -> list:
    label_to_id_converter = {f"pattern_{str(i + 1).zfill(2)}": i + 1 for i in range(24)}
    label_to_id_converter['unknown'] = 0

    # Convert labels to ids
    label_ids = [label_to_id_converter[label] for label in labels]

    return label_ids


def natural_sort(files: list) -> list:
    return natsorted(files)


def lists_diff(list1: list, list2: list) -> list:
    difference = set(list1).symmetric_difference(set(list2))

    return list(difference)
