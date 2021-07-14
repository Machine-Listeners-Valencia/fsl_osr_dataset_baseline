import pickle

import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import utils
from feature_extraction import transfer_learning
from train import train


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


def generate_embeddings():
    cwd_path = Path.cwd()

    # Input files
    csv_path = utils.recover_csv_meta_path()

    # Output paths
    features_path, l3_path, yamnet_path = utils.generate_features_paths()

    # Analysis params
    modes = ['full', 'trios']
    extractors = ['yamnet']

    for mode in modes:
        if mode == 'full':
            for extractor in extractors:
                if extractor == 'l3':
                    model = transfer_learning.AudioL3()
                    storing_path = l3_path / 'full'
                    embedding_size = 512
                if extractor == 'yamnet':
                    model = transfer_learning.YamNet()
                    storing_path = yamnet_path / 'full'
                    embedding_size = 1024
                for csv_file in sorted(csv_path.iterdir()):
                    print(f"Extracting {csv_file.stem}")
                    audios_df = pd.read_csv(csv_file)
                    training_folds = list(np.unique(audios_df['fold'].values))

                    for i in tqdm(range(len(training_folds))):
                        fold_audios_df = audios_df[audios_df['fold'] == i+1]
                        fold_audios_df = fold_audios_df.reset_index(drop=True)

                        embeddings = np.empty([fold_audios_df.shape[0], embedding_size])
                        for j, row in fold_audios_df.iterrows():
                            audio_emb = model.get_embedding(row['filename'])
                            embeddings[j, :] = audio_emb

                        fold_features = {'features': embeddings,
                                         'labels': utils.convert_to_one_hot(fold_audios_df['target'].values.tolist())}

                        storing_file = storing_path / f"{csv_file.stem}"
                        if not storing_file.is_dir():
                            storing_file.mkdir(parents=True)
                        storing_file = storing_file / f"training_fold_{i+1}.pkl"
                        pickle.dump(fold_features, open(storing_file, 'wb'))


if __name__ == "__main__":
    configuration = yaml.safe_load(open('configuration.yaml'))
    feature_extractors = configuration['feature_extractors']
    training_modes = configuration['training_modes']
    train_parameters = configuration['train_parameters']
    # prepare_splits()
    # generate_embeddings()
    train.do_training(training_modes, feature_extractors, train_parameters)
