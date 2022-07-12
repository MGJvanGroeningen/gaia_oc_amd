import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

from gaia_oc_amd.data_preparation.io import load_sets


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
    
    
def global_feature_mean_and_std(data_dir, cluster_names, training_features):
    """Calculates the global mean and standard deviation of the training features. The mean and standard deviation
    are taken over the combined sources from the cone searches of the given clusters.

    Args:
        data_dir (str): Path to the directory containing the data
        cluster_names (str, list): Names of the clusters for which the sources are combined
        training_features (str, list): List containing the training feature/column names

    Returns:
        global_mean (float, array): The global mean of each training feature
        global_std (float, array): The global standard deviation of each training feature

    """
    means = []
    stds = []

    # Store means and standard deviations of the training features for each cluster
    for cluster_name in tqdm(cluster_names, total=len(cluster_names), desc='Calculating means and stds...'):
        cluster_dir = os.path.join(data_dir, 'clusters', cluster_name)
        members, candidates, non_members, _ = load_sets(cluster_dir)
        all_sources = pd.concat((candidates, non_members))[training_features].to_numpy()
        means.append(np.mean(all_sources, axis=0))
        stds.append(np.std(all_sources, axis=0))

    means = np.array(means)
    stds = np.array(stds)

    # Calculate the global feature means and standard deviations
    global_mean = np.mean(means, axis=0)
    global_std = np.sqrt(np.mean(stds ** 2 + (means - global_mean) ** 2, axis=0))
    return global_mean, global_std


def select_train_val_indices(sources, validation_fraction, max_sources, seed):
    """First divides sources in either training or validation sources. Then, a number of source indices
    is sampled from both sets of sources to determine the sources used for the actual training and validation dataset.
    The number of samples for each set corresponds to the validation fraction.

    Args:
        sources (float, array): Sources to be divided
        validation_fraction (float): Fraction of the sources used for the validation set
        max_sources (int): The maximum number of examples per cluster included in the datasets.
        seed (int): The random seed that determines which sources are selected for the training set
            and the validation set.

    Returns:
        train_indices (int, array): The global mean of each training feature
        val_indices (int, array): The global standard deviation of each training feature

    """
    n_sources = len(sources)

    n_val_sources = int(n_sources * validation_fraction)

    max_val_sources = int(max_sources * validation_fraction)
    max_train_sources = max_sources - max_val_sources

    all_indices = np.arange(n_sources)

    np.random.seed(seed)

    # Determine train and val sources
    all_val_indices = np.random.choice(all_indices, n_val_sources, replace=False)
    all_train_indices = np.delete(all_indices, all_val_indices, axis=0)

    # Sample a maximum number of sources for the corresponding datasets
    val_indices = np.random.choice(all_val_indices, max_val_sources, replace=True)
    train_indices = np.random.choice(all_train_indices, max_train_sources, replace=True)
    return train_indices, val_indices


def deep_sets_datasets(positive_examples, negative_examples, validation_fraction=0.3, max_pos_examples=50, 
                       max_neg_examples=500, size_support_set=5, seed=42):
    """Constructs training and validation dataset instances for a deep sets model. These instances consist
    of a object to classify (either a positive or negative example) and a support set of positive examples.

    Args:
        positive_examples (float, array): Array containing normalized features of positive examples 
            with dimensions [n_pos_examples, n_features]
        negative_examples (float, array): Array containing normalized features of negative examples 
            with dimensions [n_neg_examples, n_features]
        validation_fraction (float): Fraction of the data used for the validation set
        max_pos_examples (int): The maximum number of positive examples included in the datasets. 
            If max_pos_examples=50 and validation_fraction=0.3, then the training dataset contains 
            35 positive instances and the validation dataset 15.
        max_neg_examples (int): The maximum number of negative examples included in the datasets.
        size_support_set (int): The number of positive examples in the support set
        seed (int): The random seed that determines which examples are selected for the training set
            and the validation set.

    Returns:
        train_dataset (DataLoader): The training dataset
        val_dataset (DataLoader): The validation dataset

    """
    train_dataset = []
    val_dataset = []

    # Selection of training and validation examples
    train_pos_example_indices, val_pos_example_indices = select_train_val_indices(positive_examples,
                                                                                  validation_fraction,
                                                                                  max_pos_examples, seed)
    train_neg_example_indices, val_neg_example_indices = select_train_val_indices(negative_examples,
                                                                                  validation_fraction,
                                                                                  max_neg_examples, seed)

    positive_label = [0, 1]
    negative_label = [1, 0]

    # Concatenation of a support set for each object to classify
    for dataset, pos_example_indices, neg_example_indices in zip([train_dataset, val_dataset],
                                                                 [train_pos_example_indices, val_pos_example_indices],
                                                                 [train_neg_example_indices, val_neg_example_indices]):
        for pos_example_to_classify_idx in pos_example_indices:
            pos_example_to_classify = positive_examples[pos_example_to_classify_idx]

            support_pos_examples = np.delete(positive_examples, pos_example_to_classify_idx, axis=0)
            model_input = concat_support_set(pos_example_to_classify, support_pos_examples, size_support_set)

            dataset.append((model_input, positive_label))

        for neg_example_to_classify_idx in neg_example_indices:
            neg_example_to_classify = negative_examples[neg_example_to_classify_idx]

            model_input = concat_support_set(neg_example_to_classify, positive_examples, size_support_set)

            dataset.append((model_input, negative_label))
    return train_dataset, val_dataset


def multi_cluster_deep_sets_datasets(data_dir, clusters, training_features, training_feature_means,
                                     training_feature_stds, validation_fraction=0.3, max_members=50,
                                     max_non_members=500, size_support_set=5, seed=42):
    """Function for creating a training and validation dataset from a list of clusters, 
    using their members and non-members. These datasets can be used to train a Deep Sets model.

    Args:
        data_dir (str): Path to the directory containing the data
        clusters (str, list): Names of the clusters for which datasets are to be created
        training_features (str, list): List containing the training feature/column names
        training_feature_means (float, array): Means of the training features (used for normalization)
        training_feature_stds (float, array): Standard deviation of the training features (used for normalization)
        validation_fraction (float): Fraction of the data used for the validation set
        max_members (int): The maximum number of members per cluster included in the datasets. 
        max_non_members (int): The maximum number of non-members per cluster included in the datasets.
        size_support_set (int): The number of members in the support set
        seed (int): The random seed that determines which sources are selected for the training set
            and the validation set.

    Returns:
        train_dataset (DataLoader): The training dataset
        val_dataset (DataLoader): The validation dataset

    """
    train_dataset = []
    val_dataset = []

    for cluster_name in tqdm(clusters, total=len(clusters), desc='Creating cluster(s) dataset'):
        cluster_dir = os.path.join(data_dir, 'clusters', cluster_name)
        members, candidates, non_members, _ = load_sets(cluster_dir)

        # Normalization
        members = (members[training_features].to_numpy() - training_feature_means) / training_feature_stds
        non_members = (non_members[training_features].to_numpy() - training_feature_means) / training_feature_stds

        train_data, val_data = deep_sets_datasets(members, non_members, validation_fraction=validation_fraction,
                                                  max_pos_examples=max_members, max_neg_examples=max_non_members,
                                                  size_support_set=size_support_set, seed=seed)

        train_dataset += train_data
        val_dataset += val_data

    return train_dataset, val_dataset


def deep_sets_eval_dataset(eval_sources, members, training_features, training_feature_means, training_feature_stds,
                           size_support_set=5):
    """Creates a dataset of sources which are to be evaluated by the model.

    Args:
        eval_sources (Dataframe): Sources to be evaluated by the model as member or non-member
        members (Dataframe): Dataframe containing member sources
        training_features (str, list): List containing the training feature/column names
        training_feature_means (float, array): Means of the training features (used for normalization)
        training_feature_stds (float, array): Standard deviation of the training features (used for normalization)
        size_support_set (int): The number of members in the support set

    Returns:
        train_dataset (DataLoader): The training dataset
        val_dataset (DataLoader): The validation dataset

    """
    # Normalization
    eval_sources = (eval_sources[training_features].to_numpy() - training_feature_means) / training_feature_stds
    support_members = (members[training_features].to_numpy() - training_feature_means) / training_feature_stds

    dataset = []
    for source_to_classify in eval_sources:
        model_input = concat_support_set(source_to_classify, support_members, size_support_set)
        dataset.append(model_input[None, ...])

    dataset = torch.FloatTensor(np.concatenate(dataset))
    return dataset
