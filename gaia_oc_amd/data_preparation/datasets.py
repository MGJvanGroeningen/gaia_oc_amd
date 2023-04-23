import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

from gaia_oc_amd.io import load_sets, load_cluster
from gaia_oc_amd.data_preparation.cluster import Cluster


def property_mean_and_std(data_dir, cluster_names, source_properties=None, cluster_properties=None):
    """Calculates the mean and standard deviation of a list of properties. The mean and standard deviation
    are taken over the combined sources from the cone searches of the given clusters.

    Args:
        data_dir (str): Path to the directory containing the data
        cluster_names (str, list): Names of the clusters for which the sources are combined
        source_properties (str, list): List containing the source property names
        cluster_properties (str, list): List containing the cluster property names

    Returns:
        mean (float, array): The mean of each property
        std (float, array): The standard deviation of each property

    """
    source_means = []
    source_stds = []

    cluster_means = []
    cluster_stds = []

    cluster_props = []

    # Store means and standard deviations of the training features for each cluster
    for cluster_name in tqdm(cluster_names, total=len(cluster_names), desc='Calculating means and stds...'):
        cluster_dir = os.path.join(data_dir, cluster_name)

        if source_properties is not None:
            _, candidates, non_members, _ = load_sets(cluster_dir)
            all_sources = pd.concat((candidates, non_members))[source_properties].to_numpy()
            source_means.append(np.mean(all_sources, axis=0))
            source_stds.append(np.std(all_sources, axis=0))
        if cluster_properties is not None:
            cluster_params = load_cluster(cluster_dir)
            cluster = Cluster(cluster_params)
            cluster_props.append([getattr(cluster, feature) for feature in cluster_properties])

    # Calculate the global feature means and standard deviations
    if source_properties is not None:
        source_means = np.mean(np.array(source_means), axis=0)
        source_stds = np.sqrt(np.mean(np.array(source_stds) ** 2 + (np.array(source_means) - source_means) ** 2,
                                      axis=0))
    if cluster_properties is not None:
        if len(cluster_names) == 1:
            cluster_means = np.mean(np.array(cluster_props), axis=0)
            cluster_stds = np.mean(np.array(cluster_props), axis=0)
        else:
            cluster_means = np.mean(np.array(cluster_props), axis=0)
            cluster_stds = np.std(np.array(cluster_props), axis=0)

    means = {'source': np.array(source_means), 'cluster': np.array(cluster_means)}
    stds = {'source': np.array(source_stds), 'cluster': np.array(cluster_stds)}
    return means, stds


def concat_support_set(instances_to_classify, support_instances, size_support_set, global_features=None,
                       to_classify_indices=None):
    """Concatenates a support set to the (tiled) instance to classify in order to create
    the input for the deep sets model.

    Args:
        instances_to_classify (float, array): The instances that are to be classified (n_instances, n_source_features)
        support_instances (float, array): The instances from which to create the support set (n_support_instances,
            n_source_features)
        size_support_set (int): The number of instances in the support set
        global_features (float, array): Global features to optionally add to every instance to classify
            (n_global_features)
        to_classify_indices (int, array): Indices of the instances to classify in the support instances. If these
            are supplied, they are excluded from the support set for the corresponding instances to classify.
            (n_instances)

    Returns:
        model_input (float, array): Input instance for the deep sets model

    """
    n_to_classify = instances_to_classify.shape[0]
    n_support = support_instances.shape[0]
    if size_support_set > n_support:
        raise ValueError(f"The size of the support set '{size_support_set}' cannot be greater than "
                         f"the number of support set instances '{n_support}'.")

    if to_classify_indices is not None:
        n_sample = n_support - 1
    else:
        n_sample = n_support

    if n_support > 300:
        support_set_ids = np.stack([np.random.choice(n_sample, size_support_set, replace=False)
                                    for _ in range(n_to_classify)])
    else:
        support_set_ids = np.random.rand(n_to_classify, n_sample).argsort(1)[:, :size_support_set]

    if to_classify_indices is not None:
        support_ids = np.arange(n_support)
        tiled_support_ids = np.tile(support_ids, (n_to_classify, 1))
        tiled_to_classify_ids = np.tile(to_classify_indices, (n_support, 1)).swapaxes(0, 1)
        allowed = tiled_support_ids != tiled_to_classify_ids
        allowed_support_ids = tiled_support_ids[allowed].reshape((n_to_classify, n_support - 1))
        support_set_ids = np.take_along_axis(allowed_support_ids, support_set_ids, 1)

    support_sets = support_instances[support_set_ids]

    if global_features is not None:
        tiled_global_features = np.tile(global_features, (n_to_classify, 1))
        instances_to_classify = np.concatenate((instances_to_classify, tiled_global_features), axis=-1)

    tiled_instances_to_classify = np.tile(instances_to_classify, (size_support_set, 1, 1)).swapaxes(0, 1)
    model_inputs = np.concatenate((tiled_instances_to_classify, support_sets), axis=-1)
    return model_inputs


class MultiClusterDeepSetsDataset(Dataset):
    """A Dataset child class for the model to train on. Returns the data as torch.FloatTensor
    when used as an iterator.

    Args:
        data_dir (str): Path to the directory containing the data
        cluster_names (str, list): Names of the clusters for which datasets are to be created
        source_feature_names (str, list): List containing the training feature/column names
        source_feature_means (float, array): Means of the training features (used for normalization)
        source_feature_stds (float, array): Standard deviation of the training features (used for normalization)
        cluster_feature_names (str, list): Labels of the cluster features
        cluster_feature_means (float, array): Mean values of the cluster features (normalization)
        cluster_feature_stds (float, array): Standard deviation of the cluster features (normalization)
        n_pos_duplicates (int): Number of times a positive example is included in the dataset
            (with different support set)
        neg_pos_ratio (int): Number of negative examples included in the dataset per positive example
        size_support_set (int): The number of members in the support set
        n_min_members (int): The required number of training members for a cluster to be included in the dataset
        seed (int): The random seed that determines which sources are selected for the training set
            and the validation set.

    """
    def __init__(self, data_dir, cluster_names, source_feature_names, source_feature_means=None,
                 source_feature_stds=None, cluster_feature_names=None, cluster_feature_means=None,
                 cluster_feature_stds=None, n_pos_duplicates=2, neg_pos_ratio=5, size_support_set=10,
                 n_min_members=15, seed=42):
        # Cluster names
        self.cluster_names = cluster_names

        # Feature names
        self.source_feature_names = source_feature_names
        self.cluster_feature_names = cluster_feature_names
        self.feature_names = source_feature_names + cluster_feature_names

        # Normalization
        self.source_feature_means = source_feature_means
        self.source_feature_stds = source_feature_stds
        self.cluster_feature_means = cluster_feature_means
        self.cluster_feature_stds = cluster_feature_stds

        # Dataset hyperparamters
        self.n_pos_duplicates = n_pos_duplicates
        self.neg_pos_ratio = neg_pos_ratio
        self.size_support_set = size_support_set
        self.n_min_members = n_min_members
        self.seed = seed

        # Dataset
        self.dataset = multi_cluster_deep_sets_dataset(data_dir, cluster_names, source_feature_names,
                                                       source_feature_means=source_feature_means,
                                                       source_feature_stds=source_feature_stds,
                                                       cluster_feature_names=cluster_feature_names,
                                                       cluster_feature_means=cluster_feature_means,
                                                       cluster_feature_stds=cluster_feature_stds,
                                                       n_pos_duplicates=n_pos_duplicates, neg_pos_ratio=neg_pos_ratio,
                                                       size_support_set=size_support_set, n_min_members=n_min_members,
                                                       seed=seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        model_input, label = self.dataset[item]
        return torch.FloatTensor(model_input), torch.FloatTensor(label)


def deep_sets_dataset(positive_examples, negative_examples, global_features=None, n_pos_duplicates=1, neg_pos_ratio=1,
                      size_support_set=10, seed=42):
    """Constructs dataset instances for a deep sets model. These instances consist of an object to classify
    (either a positive or negative example) and a support set of positive examples.

    Args:
        positive_examples (float, array): Array containing normalized features of positive examples 
            with dimensions [n_pos_examples, n_features]
        negative_examples (float, array): Array containing normalized features of negative examples 
            with dimensions [n_neg_examples, n_features]
        global_features (float, array): Global features to add to every support example
        n_pos_duplicates (int): Number of times a positive example is included in the dataset
            (with different support set)
        neg_pos_ratio (int): Number of negative examples included in the dataset per positive example
        size_support_set (int): The number of positive examples in the support set
        seed (int): The random seed that determines which examples are selected for the training set
            and the validation set.

    Returns:
        dataset (tuple, list): A deep sets dataset

    """
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

    dataset = []

    # Concatenation of a support set for each object to classify
    positive_model_inputs = concat_support_set(positive_examples[pos_example_indices], positive_examples,
                                               size_support_set, global_features=global_features,
                                               to_classify_indices=pos_example_indices)
    for positive_model_input in positive_model_inputs:
        dataset.append((positive_model_input, positive_label))

    negative_model_inputs = concat_support_set(negative_examples[neg_example_indices], positive_examples,
                                               size_support_set, global_features=global_features)
    for negative_model_input in negative_model_inputs:
        dataset.append((negative_model_input, negative_label))
    return dataset


def multi_cluster_deep_sets_dataset(data_dir, cluster_names, source_feature_names, source_feature_means=None,
                                    source_feature_stds=None, cluster_feature_names=None, cluster_feature_means=None,
                                    cluster_feature_stds=None, n_pos_duplicates=2, neg_pos_ratio=5, size_support_set=10,
                                    n_min_members=15, seed=42):
    """Function for creating a dataset from a list of clusters, using their members and non-members.
    This dataset can be used to train a Deep Sets model.

    Args:
        data_dir (str): Path to the directory containing the data
        cluster_names (str, list): Names of the clusters for which datasets are to be created
        source_feature_names (str, list): List containing the training feature/column names
        source_feature_means (float, array): Means of the training features (used for normalization)
        source_feature_stds (float, array): Standard deviation of the training features (used for normalization)
        cluster_feature_names (str, list): Labels of the cluster features
        cluster_feature_means (float, array): Mean values of the cluster features (normalization)
        cluster_feature_stds (float, array): Standard deviation of the cluster features (normalization)
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

    for cluster_name in tqdm(cluster_names, total=len(cluster_names), desc='Creating cluster(s) dataset'):
        cluster_dir = os.path.join(data_dir, cluster_name)

        # Load cluster data
        members, _, non_members, _ = load_sets(cluster_dir)
        cluster_params = load_cluster(cluster_dir)
        cluster = Cluster(cluster_params)

        # Optionally add cluster features
        if cluster_feature_names is not None:
            cluster_features = np.array([getattr(cluster, feature) for feature in cluster_feature_names])
            if cluster_feature_means is not None and cluster_feature_stds is not None:
                cluster_features = (cluster_features - cluster_feature_means) / cluster_feature_stds
        else:
            cluster_features = None

        # Only include clusters which have more than a certain amount number of members.
        if len(members) >= n_min_members:
            members = members[source_feature_names].to_numpy()
            non_members = non_members[source_feature_names].to_numpy()

            # Normalization
            if source_feature_means is not None and source_feature_stds is not None:
                members = (members - source_feature_means) / source_feature_stds
                non_members = (non_members - source_feature_means) / source_feature_stds

            # Create a deep sets dataset for the cluster
            cluster_dataset = deep_sets_dataset(members, non_members, global_features=cluster_features,
                                                n_pos_duplicates=n_pos_duplicates, neg_pos_ratio=neg_pos_ratio,
                                                size_support_set=size_support_set, seed=seed)

            multi_cluster_dataset += cluster_dataset

    return multi_cluster_dataset


def deep_sets_eval_dataset(eval_data, support_examples, global_features=None, size_support_set=10):
    """Creates a dataset of sources which are to be evaluated by the model.

    Args:
        eval_data (float, array): Array containing normalized features of instances to be evaluated
            with dimensions [n_instances, n_features]
        support_examples (float, array): Array containing normalized features of instances to be used
            for the support set [n_support_instances, n_features]
        global_features (float, array): Global features to add to every support example
        size_support_set (int): The number of instances in the support set

    Returns:
        dataset (torch.FloatTensor): The evaluation dataset

    """
    dataset = concat_support_set(eval_data, support_examples, size_support_set,
                                 global_features=global_features)
    dataset = torch.FloatTensor(dataset)
    return dataset
