import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from gaia_oc_amd.data_preparation.utils import normalize, load_sets
from gaia_oc_amd.data_preparation.features import add_features


def concat_source_support(source_to_classify, members, support_set_indices, size_support_set):
    support_set_indices = np.random.choice(support_set_indices, size_support_set, replace=False)
    support_set = members[support_set_indices]

    tiled_source_to_classify = np.tile(source_to_classify, (size_support_set, 1))
    model_input = np.concatenate((support_set, tiled_source_to_classify), axis=1)
    return model_input


class DeepSetsDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        model_input, label = self.dataset[item]
        return torch.FloatTensor(model_input), torch.FloatTensor(label)


def deep_sets_datasets(data_dir, clusters, training_features, validation_fraction, max_members=50, max_non_members=500,
                       size_support_set=5, batch_size=32, seed=42):
    train_dataset = []
    val_dataset = []

    for cluster_name in tqdm(clusters, total=len(clusters), desc='Creating cluster(s) dataset'):
        _, members, candidates, non_members, _ = load_sets(data_dir, cluster_name)
        all_sources = pd.concat((members, candidates, non_members))

        members = normalize(members[training_features], all_data=all_sources[training_features]).to_numpy()
        non_members = normalize(non_members[training_features], all_data=all_sources[training_features]).to_numpy()

        n_members = len(members)
        n_non_members = len(non_members)

        n_val_members = int(n_members * validation_fraction)
        max_val_members = int(max_members * validation_fraction)

        n_val_non_members = int(n_non_members * validation_fraction)
        max_val_non_members = int(max_non_members * validation_fraction)

        np.random.seed(seed)
        val_member_indices = np.random.choice(np.arange(n_members), n_val_members, replace=False)
        val_member_indices = np.random.choice(val_member_indices, max_val_members, replace=True)

        val_non_member_indices = np.random.choice(np.arange(n_non_members), n_val_non_members, replace=False)
        val_non_member_indices = np.random.choice(val_non_member_indices, max_val_non_members, replace=True)

        train_member_indices = np.delete(np.arange(n_members), val_member_indices, axis=0)
        train_member_indices = np.random.choice(train_member_indices, max_members - max_val_members, replace=True)

        train_non_member_indices = np.delete(np.arange(n_non_members), val_non_member_indices, axis=0)
        train_non_member_indices = np.random.choice(train_non_member_indices, max_non_members - max_val_non_members,
                                                    replace=True)

        all_member_indices = list(np.arange(n_members))
        member_label = [0, 1]
        non_member_label = [1, 0]

        for dataset, member_indices, non_member_indices in zip([train_dataset, val_dataset],
                                                               [train_member_indices, val_member_indices],
                                                               [train_non_member_indices, val_non_member_indices]):
            for source_to_classify_idx in member_indices:
                source_to_classify = members[source_to_classify_idx]

                viable_support_set_indices = np.delete(all_member_indices, source_to_classify_idx, axis=0)
                model_input = concat_source_support(source_to_classify, members, viable_support_set_indices,
                                                    size_support_set)

                dataset.append((model_input, member_label))

            for source_to_classify_idx in non_member_indices:
                source_to_classify = non_members[source_to_classify_idx]

                viable_support_set_indices = all_member_indices
                model_input = concat_source_support(source_to_classify, members, viable_support_set_indices,
                                                    size_support_set)

                dataset.append((model_input, non_member_label))

    train_dataset = DataLoader(DeepSetsDataset(train_dataset), batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(DeepSetsDataset(val_dataset), batch_size=batch_size, shuffle=True)

    return train_dataset, val_dataset


def candidate_dataset(candidate_sample, sources, features, size_support_set):
    add_features([candidate_sample], features.train_feature_functions, features.train_feature_labels)

    all_data = sources.all_sources[features.train_features]
    candidate_sample = normalize(candidate_sample[features.train_features], all_data=all_data).to_numpy()
    members = normalize(sources.members.hp()[features.train_features], all_data=all_data).to_numpy()

    dataset = []
    member_indices = list(np.arange(len(members)))
    for source_to_classify in candidate_sample:
        model_input = concat_source_support(source_to_classify, members, member_indices, size_support_set)
        dataset.append(model_input[None, ...])

    return torch.FloatTensor(np.concatenate(dataset))
