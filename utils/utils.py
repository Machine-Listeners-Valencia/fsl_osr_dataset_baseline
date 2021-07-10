from typing import Tuple
import random

from pathlib import Path
import pandas as pd


def prepare_paths():
    cwd_path = Path.cwd()

    # Data paths
    data_path = cwd_path / 'data'
    metadata_path = data_path / 'meta'
    pattern_path = data_path / 'pattern_sounds'
    unwanted_path = data_path / 'unwanted_sounds'

    return data_path, metadata_path, pattern_path, unwanted_path


def folders_in_path(path: Path) -> list:
    return [folder for folder in sorted(path.iterdir()) if folder.is_dir()]


def prepare_extraction_options(pattern_path: Path) -> Tuple[list, dict, list, int, int]:
    # Labels
    pattern_folders = pattern_path.glob('**/')
    labels = [f"pattern_{str(i + 1).zfill(2)}" for i, _ in enumerate(pattern_folders)]
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


def create_configuration_csvs(pattern_files: list,
                              unknown_files: list,
                              pattern_labels: list,
                              unknown_labels: list,
                              pattern_train_folds: list,
                              unknown_train_folds: list,
                              shots: dict,
                              openness_factors: list,
                              metadata_path: Path
                              ) -> None:
    files = pattern_files + unknown_files
    labels = pattern_labels + unknown_labels
    for shot, number_of_folds in shots.items():
        train_folds = pattern_train_folds[shot] + unknown_train_folds[shot]
        for openness in openness_factors:
            if openness == 0:
                test_classes_list = []
            elif openness == 50:
                test_classes_list = ['clapping', 'door_slam', 'water', 'music', 'pots_and_pans']
                number_of_folds = number_of_folds + 1
            elif openness == 100:
                test_classes_list = ['car_horn', 'clapping', 'cough', 'door_slam', 'engine',
                                     'keyboard_tap', 'music', 'pots_and_pans', 'steps', 'water']
                number_of_folds = number_of_folds + 1

            for class_name in test_classes_list:
                indexes = match_substring(pattern_files, class_name)

                for i in indexes:
                    train_folds[i] = number_of_folds

            csv_root_path = metadata_path / 'csv'
            if not csv_root_path.is_dir():
                csv_root_path.mkdir()
            csv_path = csv_root_path / f"few_shot_k_{shot}_openness_{openness}.csv"
            metadata = {'filename': files, 'target': labels, 'fold': train_folds}
            metadata_df = pd.DataFrame(metadata, columns=['filename', 'target', 'fold'])
            metadata_df.to_csv(path_or_buf=csv_path, index=False, header=True)