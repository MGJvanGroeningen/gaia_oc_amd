import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from gaia_oc_amd.io import load_sets


def concat_support_set(source_to_classify, support_members, size_support_set):
    """Concatenates a support set of members to the (tiled) member to classify
    in order to create the input for the deep sets model.

    Args:
        source_to_classify (float, array): The source that is to be classified
        support_members (float, array): The members from which to select the support set members
        size_support_set (int): The number of members in the support set

    Returns:
        model_input (float, array): Input instance for the deep sets model

    """
    n_support_members = len(support_members)
    if size_support_set > n_support_members:
        raise ValueError(f"The size of the support set '{size_support_set}' cannot be greater than "
                         f"the number of support set members '{n_support_members}'.")

    support_set_indices = np.random.choice(n_support_members, size_support_set, replace=False)
    support_set = support_members[support_set_indices]

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


def deep_sets_dataset(positive_examples, negative_examples, n_pos_duplicates=1, neg_pos_ratio=1, size_support_set=10,
                      seed=42):
    """Constructs dataset instances for a deep sets model. These instances consist of an object to classify
    (either a positive or negative example) and a support set of positive examples.

    Args:
        positive_examples (float, array): Array containing normalized features of positive examples 
            with dimensions [n_pos_examples, n_features]
        negative_examples (float, array): Array containing normalized features of negative examples 
            with dimensions [n_neg_examples, n_features]
        n_pos_duplicates (int): Number of times a positive example is included in the dataset
            (with different support set)
        neg_pos_ratio (int): Number of negative examples included in the dataset per positive example
        size_support_set (int): The number of positive examples in the support set
        seed (int): The random seed that determines which examples are selected for the training set
            and the validation set.

    Returns:
        dataset (tuple, list): A deep sets dataset

    """
    dataset = []

    # Include all positive examples 'n_pos_duplicates' number of times
    n_positive_examples = positive_examples.shape[0]
    pos_example_indices = np.concatenate(n_pos_duplicates * [np.arange(n_positive_examples)])

    n_negative_examples = negative_examples.shape[0]
    n_negative_instances = n_positive_examples * n_pos_duplicates * neg_pos_ratio

    # Either select 'n_negative_instances' random examples from the negative examples
    # if there are fewer instances than examples, or include all examples and pad with duplicates
    # and/or randomly selected examples.
    np.random.seed(seed)
    if n_negative_instances < n_negative_examples:
        neg_example_indices = np.random.choice(np.arange(n_negative_examples), n_negative_instances, replace=False)
    else:
        additional_indices = np.random.choice(np.arange(n_negative_examples), n_negative_instances %
                                              n_negative_examples, replace=False)
        neg_example_indices = np.concatenate((n_negative_instances // n_negative_examples) *
                                             [np.arange(n_negative_examples)] + [additional_indices])

    positive_label = [0, 1]
    negative_label = [1, 0]

    # Concatenation of a support set for each object to classify
    for pos_example_to_classify_idx in pos_example_indices:
        pos_example_to_classify = positive_examples[pos_example_to_classify_idx]

        support_pos_examples = np.delete(positive_examples, pos_example_to_classify_idx, axis=0)
        model_input = concat_support_set(pos_example_to_classify, support_pos_examples, size_support_set)

        dataset.append((model_input, positive_label))

    for neg_example_to_classify_idx in neg_example_indices:
        neg_example_to_classify = negative_examples[neg_example_to_classify_idx]

        model_input = concat_support_set(neg_example_to_classify, positive_examples, size_support_set)

        dataset.append((model_input, negative_label))
    return dataset


def multi_cluster_deep_sets_dataset(data_dir, clusters, training_features, training_feature_means,
                                    training_feature_stds, n_pos_duplicates=2, neg_pos_ratio=5, size_support_set=10,
                                    n_min_members=15, seed=42):
    """Function for creating a dataset from a list of clusters, using their members and non-members.
    This dataset can be used to train a Deep Sets model.

    Args:
        data_dir (str): Path to the directory containing the data
        clusters (str, list): Names of the clusters for which datasets are to be created
        training_features (str, list): List containing the training feature/column names
        training_feature_means (float, array): Means of the training features (used for normalization)
        training_feature_stds (float, array): Standard deviation of the training features (used for normalization)
        n_pos_duplicates (int): Number of times a positive example is included in the dataset
            (with different support set)
        neg_pos_ratio (int): Number of negative examples included in the dataset per positive example
        size_support_set (int): The number of members in the support set
        n_min_members (int): The required number of training members for a cluster to be included in the dataset
        seed (int): The random seed that determines which sources are selected for the training set
            and the validation set.

    Returns:
        dataset (tuple, list): The dataset

    """
    multi_cluster_dataset = []

    for cluster_name in tqdm(clusters, total=len(clusters), desc='Creating cluster(s) dataset'):
        cluster_dir = os.path.join(data_dir, cluster_name)
        members, candidates, non_members, _ = load_sets(cluster_dir)
        if len(members) >= n_min_members:
            # Normalization
            members = (members[training_features].to_numpy() - training_feature_means) / training_feature_stds
            non_members = (non_members[training_features].to_numpy() - training_feature_means) / training_feature_stds

            cluster_dataset = deep_sets_dataset(members, non_members, n_pos_duplicates=n_pos_duplicates,
                                                neg_pos_ratio=neg_pos_ratio, size_support_set=size_support_set,
                                                seed=seed)

            multi_cluster_dataset += cluster_dataset

    return multi_cluster_dataset


def deep_sets_eval_dataset(eval_sources, support_members, training_features, training_feature_means,
                           training_feature_stds, size_support_set=10):
    """Creates a dataset of sources which are to be evaluated by the model.

    Args:
        eval_sources (Dataframe): Sources to be evaluated by the model as member or non-member
        support_members (Dataframe): Dataframe containing member sources to be used for the support set
        training_features (str, list): List containing the training feature/column names
        training_feature_means (float, array): Means of the training features (used for normalization)
        training_feature_stds (float, array): Standard deviation of the training features (used for normalization)
        size_support_set (int): The number of members in the support set

    Returns:
        dataset (torch.FloatTensor): The evaluation dataset

    """
    # Normalization
    eval_sources = (eval_sources[training_features].to_numpy() - training_feature_means) / training_feature_stds
    support_members = (support_members[training_features].to_numpy() - training_feature_means) / training_feature_stds

    dataset = []
    for source_to_classify in eval_sources:
        model_input = concat_support_set(source_to_classify, support_members, size_support_set)
        dataset.append(model_input[None, ...])

    dataset = torch.FloatTensor(np.concatenate(dataset))
    return dataset
