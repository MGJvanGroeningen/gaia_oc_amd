import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from gaia_oc_amd.data_preparation.utils import normalize, load_sets


def concat_support_set(source_to_classify, members, size_support_set):
    """Concatenates a support set of members to the (tiled) member to classify
    in order to create the input for the deep sets model.

    Args:
        source_to_classify (float, array): THe source that is to be classified
        members (float, array): The members from which to select the support set members
        size_support_set (int): The number of members in the support set

    Returns:
        model_input (float, array): Input instance for the deep sets model

    """
    support_set_indices = np.random.choice(len(members), size_support_set, replace=False)
    support_set = members[support_set_indices]

    tiled_source_to_classify = np.tile(source_to_classify, (size_support_set, 1))
    model_input = np.concatenate((tiled_source_to_classify, support_set), axis=1)
    return model_input


class DeepSetsDataset(Dataset):
    """A Dataset child class for the model to train on. Returns the data as torch.FloatTensor
    when used as an iterator.

    Args:
        dataset (list): A list of (model_input, label) tuples

    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        model_input, label = self.dataset[item]
        return torch.FloatTensor(model_input), torch.FloatTensor(label)


def deep_sets_datasets(data_dir, clusters, training_features, validation_fraction, max_members=50, max_non_members=500,
                       size_support_set=5, batch_size=32, seed=42):
    """Function for creating the training and validation dataset from the members,

    Args:
        source_to_classify (float, array): THe source that is to be classified
        members (float, array): The members from which to select the support set members
        size_support_set (int): The number of members in the support set

    Returns:
        model_input (float, array): Input instance for the deep sets model

    """
    train_dataset = []
    val_dataset = []

    means = []
    stds = []

    for cluster_name in tqdm(clusters, total=len(clusters), desc='Calculating means and stds...'):
        _, members, candidates, non_members, _ = load_sets(data_dir, cluster_name)
        all_sources = pd.concat((candidates, non_members))[training_features].to_numpy()
        means.append(np.mean(all_sources, axis=0))
        stds.append(np.std(all_sources, axis=0))

    means = np.array(means)
    stds = np.array(stds)

    global_mean = np.mean(means, axis=0)
    global_std = np.sqrt(np.mean(stds ** 2 + (means - global_mean) ** 2, axis=0))

    for cluster_name in tqdm(clusters, total=len(clusters), desc='Creating cluster(s) dataset'):
        _, members, candidates, non_members, _ = load_sets(data_dir, cluster_name)
        # all_sources = pd.concat((candidates, non_members))

        members = (members[training_features].to_numpy() - global_mean) / global_std
        non_members = (non_members[training_features].to_numpy() - global_mean) / global_std

        # members = normalize(members[training_features], all_data=all_sources[training_features]).to_numpy()
        # non_members = normalize(non_members[training_features], all_data=all_sources[training_features]).to_numpy()

        n_members = len(members)
        n_non_members = len(non_members)

        n_val_members = int(n_members * validation_fraction)
        n_val_non_members = int(n_non_members * validation_fraction)

        max_val_members = int(max_members * validation_fraction)
        max_train_members = max_members - max_val_members
        max_val_non_members = int(max_non_members * validation_fraction)
        max_train_non_members = max_non_members - max_val_non_members

        all_member_indices = np.arange(n_members)
        all_non_member_indices = np.arange(n_non_members)

        np.random.seed(seed)
        all_val_member_indices = np.random.choice(all_member_indices, n_val_members, replace=False)
        all_train_member_indices = np.delete(all_member_indices, all_val_member_indices, axis=0)
        all_val_non_member_indices = np.random.choice(all_non_member_indices, n_val_non_members, replace=False)
        all_train_non_member_indices = np.delete(all_non_member_indices, all_val_non_member_indices, axis=0)

        val_member_indices = np.random.choice(all_val_member_indices, max_val_members, replace=True)
        train_member_indices = np.random.choice(all_train_member_indices, max_train_members, replace=True)
        val_non_member_indices = np.random.choice(all_val_non_member_indices, max_val_non_members, replace=True)
        train_non_member_indices = np.random.choice(all_train_non_member_indices, max_train_non_members, replace=True)

        member_label = [0, 1]
        non_member_label = [1, 0]

        for dataset, member_indices, non_member_indices in zip([train_dataset, val_dataset],
                                                               [train_member_indices, val_member_indices],
                                                               [train_non_member_indices, val_non_member_indices]):
            for member_to_classify_idx in member_indices:
                member_to_classify = members[member_to_classify_idx]

                support_members = np.delete(members, member_to_classify_idx, axis=0)
                model_input = concat_support_set(member_to_classify, support_members, size_support_set)

                dataset.append((model_input, member_label))

            for non_member_to_classify_idx in non_member_indices:
                non_member_to_classify = non_members[non_member_to_classify_idx]

                model_input = concat_support_set(non_member_to_classify, members, size_support_set)

                dataset.append((model_input, non_member_label))

    train_dataset = DataLoader(DeepSetsDataset(train_dataset), batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(DeepSetsDataset(val_dataset), batch_size=batch_size, shuffle=True)

    return train_dataset, val_dataset


def deep_sets_eval_dataset(eval_sources, members, size_support_set):
    dataset = []
    for source_to_classify in eval_sources:
        model_input = concat_support_set(source_to_classify, members, size_support_set)
        dataset.append(model_input[None, ...])

    dataset = torch.FloatTensor(np.concatenate(dataset))
    return dataset
