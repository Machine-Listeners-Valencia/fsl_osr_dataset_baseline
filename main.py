from pathlib import Path
import pandas as pd

from utils import utils


def prepare_splits():
    cwd_path = Path.cwd()

    # Recover paths
    data_path, metadata_path, pattern_path, unknown_path = utils.prepare_paths()
    pattern_folders = utils.folders_in_path(pattern_path)
    unknown_folders = utils.folders_in_path(unknown_path)

    # Recover options
    options = utils.prepare_extraction_options(vars()['pattern_path'])
    tags, shots, openness_factors, number_of_classes, number_of_folder_files = options

    # Prepare list of files, labels and train folds for pattern files
    pattern_files, pattern_labels = utils.prepare_files_and_labels(pattern_path, tags[1:])
    pattern_train_folds = utils.prepare_train_folds(shots, number_of_folder_files, pattern_folders)

    # Prepare list of files, labels and train folds for unknown files
    unknown_files, unknown_labels = utils.prepare_unknown_files_and_labels(unknown_path)
    unknown_train_folds = utils.prepare_train_folds(shots, number_of_folder_files, unknown_folders)

    utils.create_configuration_csvs(pattern_files, unknown_files, pattern_labels, unknown_labels, pattern_train_folds,
                                    unknown_train_folds, shots, openness_factors, metadata_path)


if __name__ == "__main__":
    prepare_splits()
