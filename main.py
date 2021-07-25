import pickle

import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import utils
from feature_extraction import embeddings
from train import train


def prepare_splits():
    cwd_path = Path.cwd()

    # Recover paths
    data_path, metadata_path, pattern_path, unknown_path = utils.prepare_paths()
    pattern_folders = utils.folders_in_path(pattern_path)
    unknown_folders = utils.folders_in_path(unknown_path)

    # Recover options
    options = utils.prepare_extraction_options(pattern_path)
    tags, shots, openness_factors, number_of_classes, number_of_folder_files = options

    # Prepare list of files, labels and train folds for pattern files
    pattern_files, pattern_labels = utils.prepare_files_and_labels(pattern_path, tags[1:])
    pattern_train_folds = utils.prepare_train_folds(shots, number_of_folder_files, pattern_folders)
    pattern_trios_indexes = utils.prepare_pattern_trios_indexes(number_of_folder_files, number_of_classes)

    # Prepare list of files, labels and train folds for unknown files
    unknown_files, unknown_labels = utils.prepare_unknown_files_and_labels(unknown_path)
    unknown_train_folds = utils.prepare_train_folds(shots, number_of_folder_files, unknown_folders)
    unknown_trios_indexes = utils.prepare_unknown_trios_indexes(unknown_files)

    utils.create_configuration_csvs(pattern_files, unknown_files, pattern_labels, unknown_labels, pattern_train_folds,
                                    unknown_train_folds, pattern_trios_indexes, unknown_trios_indexes,
                                    shots, openness_factors, metadata_path)


if __name__ == "__main__":
    # Load configuration parameters
    configuration = yaml.safe_load(open('configuration.yaml'))
    feature_extractors = configuration['feature_extractors']
    training_modes = configuration['training_modes']
    train_parameters = configuration['train_parameters']
    stages = configuration['stages']

    # Run specified stages
    if stages['splits']:
        prepare_splits()
    if stages['embeddings']:
        embeddings.generate_embeddings(training_modes, feature_extractors)
    if stages['training']:
        train.do_training(training_modes, feature_extractors, train_parameters)