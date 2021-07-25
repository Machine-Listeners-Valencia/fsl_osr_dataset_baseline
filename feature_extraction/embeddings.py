import pickle

from pathlib import Path
from typing import Union, Tuple

from tqdm import tqdm
import numpy as np
import pandas as pd

from utils import utils
from feature_extraction import transfer_learning


def generate_embeddings(modes: list, extractors: list):
    # Input files
    csv_path = utils.recover_csv_meta_path()

    # Output paths
    features_path, l3_path, yamnet_path = utils.generate_features_paths()

    for mode in modes:
        if mode == 'full':
            for extractor in extractors:
                model, storing_path, embedding_size = select_feature_extractor(extractor, mode, l3_path, yamnet_path)
                for csv_file in sorted(csv_path.iterdir()):
                    print(f"Extracting {csv_file.stem}")
                    audios_df = pd.read_csv(csv_file)
                    training_folds = list(np.unique(audios_df['fold'].values))

                    # Extract and store training folds
                    extract_folds(audios_df, model, training_folds, embedding_size, csv_file.stem, storing_path)

        elif mode == 'trios':
            number_of_trios = 8
            for extractor in extractors:
                model, storing_path, embedding_size = select_feature_extractor(extractor, mode, l3_path, yamnet_path)
                for csv_file in sorted(csv_path.iterdir()):
                    print(f"Extracting {csv_file.stem}")
                    audios_df = pd.read_csv(csv_file)
                    training_folds = list(np.unique(audios_df['fold'].values))

                    for j in range(number_of_trios):
                        trio_df = audios_df[(audios_df['trio_index'] == j) | (audios_df['trio_index'] == 8)]
                        storing_trio_path = storing_path / f"trio_{j}"

                        # Extract and store training folds
                        extract_folds(trio_df, model, training_folds, embedding_size,
                                      csv_file.stem, storing_trio_path, mode)


def extract_folds(df: pd.DataFrame,
                  model: Union[transfer_learning.YamNet, transfer_learning.AudioL3],
                  training_folds: int,
                  embedding_size: int,
                  csv_name: str,
                  storing_root_path: Path,
                  mode: str = 'full'
                  ) -> None:
    for i in tqdm(training_folds):
        # Select audios of specific fold
        fold_audios_df = df[df['fold'] == i]
        fold_audios_df = fold_audios_df.reset_index(drop=True)

        # Extract embeddings
        embeddings = np.empty([fold_audios_df.shape[0], embedding_size])
        for j, row in fold_audios_df.iterrows():
            audio_emb = model.get_embedding(row['filename'])
            embeddings[j, :] = audio_emb

        # Prepare training structure (features + labels)
        fold_features = {'features': embeddings,
                         'labels': utils.convert_to_one_hot(fold_audios_df['target'].values.tolist(), mode)}

        # Store training structure fold into specific path
        storing_file = storing_root_path / f"{csv_name}"
        if not storing_file.is_dir():
            storing_file.mkdir(parents=True)
        storing_file = storing_file / f"training_fold_{i}.pkl"
        pickle.dump(fold_features, open(storing_file, 'wb'))


def select_feature_extractor(extractor: str,
                             training_mode: str,
                             l3_path: Path,
                             yamnet_path: Path
                             ) -> Tuple[Union[transfer_learning.AudioL3, transfer_learning.YamNet, Path, int]]:
    if extractor == 'l3':
        model = transfer_learning.AudioL3()
        storing_path = l3_path / training_mode
        embedding_size = 512
    if extractor == 'yamnet':
        model = transfer_learning.YamNet()
        storing_path = yamnet_path / training_mode
        embedding_size = 1024

    return model, storing_path, embedding_size
