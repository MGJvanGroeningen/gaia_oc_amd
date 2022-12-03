import os
import numpy as np
from tqdm import tqdm
from torch.nn.functional import softmax

from gaia_oc_amd.data_preparation.cluster import Cluster
from gaia_oc_amd.data_preparation.datasets import deep_sets_eval_dataset
from gaia_oc_amd.data_preparation.features import add_features
from gaia_oc_amd.io import load_model, load_hyper_parameters, load_cluster, load_sets


def means_and_covariance_matrix(sources, properties):
    """Determines the means and covariance matrix for a set of sources of a number of properties.

    Args:
        sources (Dataframe): Dataframe containing sources for which we want to determine
            the means and covariance matrix
        properties (str, list): Labels of the properties for which to determine the means and covariances

    Returns:
        means (float, array): Array with mean properties of the sources,
            dimensions (n_sources, n_properties)
        cov_matrices (float, array): Array with the property covariances of the sources,
            dimensions (n_sources, n_properties, n_properties)

    """
    n_sources = len(sources)
    n_properties = len(properties)
    source_columns = sources.columns.to_list()

    property_errors = [prop + '_error' for prop in properties]

    means = np.zeros((n_properties, n_sources))
    cov_matrices = np.zeros((n_properties, n_properties, n_sources))

    for i in range(n_properties):
        means[i] = sources[properties[i]]
        for j in range(i, n_properties):
            if i == j:
                sigma = sources[property_errors[i]].to_numpy()
                cov_matrices[i, j] = sigma ** 2
            else:
                property_corr_label1 = properties[i] + '_' + properties[j] + '_corr'
                property_corr_label2 = properties[j] + '_' + properties[i] + '_corr'
                if property_corr_label1 in source_columns:
                    corr = sources[property_corr_label1].to_numpy()
                elif property_corr_label2 in source_columns:
                    corr = sources[property_corr_label2].to_numpy()
                else:
                    corr = 0
                sigma_i = sources[property_errors[i]].to_numpy()
                sigma_j = sources[property_errors[j]].to_numpy()
                cov = corr * sigma_i * sigma_j
                cov_matrices[i, j] = cov
                cov_matrices[j, i] = cov
    return means, cov_matrices


def sample_sources(sources, sample_properties, n_samples=1):
    """Creates an array of sampled properties for a set of sources.

    Args:
        sources (Dataframe): Dataframe containing sources for which we want to sample some properties
        sample_properties (str, list): Labels of the properties we want to sample
        n_samples (int): The number of samples

    Returns:
        samples (float, array): Array with the sampled properties of the sources,
            dimensions (n_samples, n_sources, n_properties)

    """
    n_sources = len(sources)
    n_properties = len(sample_properties)
    samples = np.zeros((n_sources, n_samples, n_properties))

    # Determine the means and covariance matrix of the astrometric and photometric properties of the sources
    property_means, property_cov_matrices = means_and_covariance_matrix(sources, sample_properties)

    for i in range(n_sources):
        samples[i] = np.random.multivariate_normal(mean=property_means[:, i], cov=property_cov_matrices[:, :, i],
                                                   size=n_samples)
    samples = np.swapaxes(samples, 0, 1)
    return samples


def predict_membership(dataset, model, batch_size=1024):
    """Returns a boolean value for each source in the dataset, where True indicates that the model
    thinks that the source is a member.

    Args:
        dataset (FloatTensor): Deep sets dataset, has dimensions (n_data, size_support_set, 2 * n_features)
        model (nn.Module): Deep sets model
        batch_size (float, array): Size of the batches in which the dataset is divided

    Returns:
        predictions (bool, array): Array of membership predictions for the sources in the dataset

    """
    n_batches = dataset.size(0) // batch_size + 1
    predictions = []
    start = 0

    for _ in range(n_batches):
        batch_data = dataset[start:start + batch_size]
        batch_output = model(batch_data)
        batch_member_probabilities = softmax(batch_output, 1).detach().numpy()[:, 1]
        batch_predictions = batch_member_probabilities > 0.5
        predictions.append(batch_predictions)
        start += batch_size

    predictions = np.concatenate(predictions)
    return predictions


def calculate_candidate_probs(cluster_dir, model_dir, n_samples=40, fast_mode=True, seed=42):
    """Returns the membership probabilities of a set of sources, based on the model predictions
    for a number of sampled properties of these sources.

    Args:
        cluster_dir (str): Directory where the cluster data is stored
        model_dir (str): Directory where the model data is stored
        n_samples (int): The number of samples
        fast_mode (bool): If True, use a faster but more memory intensive method, which might crash for
            many (>~10^5) sources.
        seed (int): The random seed, which determines the samples

    Returns:
        probabilities (float, array): Array of membership probabilities for the to-be-evaluated sources

    """
    # Load the necessary datasets (i.e. cluster, sources)
    cluster_params = load_cluster(cluster_dir)
    cluster = Cluster(cluster_params)
    members, candidates, _, _ = load_sets(cluster_dir)

    # Load the (trained) model
    model = load_model(model_dir)
    model.eval()

    # Load the dataset hyperparameters
    hyper_parameters = load_hyper_parameters(model_dir)

    source_feature_names = hyper_parameters['source_features']
    source_feature_means = np.array(hyper_parameters['source_feature_means'])
    source_feature_stds = np.array(hyper_parameters['source_feature_stds'])

    cluster_feature_names = hyper_parameters['cluster_features']
    cluster_feature_means = np.array(hyper_parameters['cluster_feature_means'])
    cluster_feature_stds = np.array(hyper_parameters['cluster_feature_stds'])

    size_support_set = hyper_parameters['size_support_set']

    # Sample astrometric and photometric properties of the sources
    np.random.seed(seed)
    sample_properties = ['parallax', 'pmra', 'pmdec', 'phot_g_mean_mag', cluster.isochrone_colour]
    samples = sample_sources(candidates, sample_properties=sample_properties, n_samples=n_samples)
    candidates_sample_predictions = np.zeros((n_samples, len(candidates)))

    update_f_pm = 'f_pm' in source_feature_names
    update_f_plx = 'f_plx' in source_feature_names
    update_f_iso = 'f_c' in source_feature_names or 'f_g' in source_feature_names

    # Determine the cluster features if they were used for training
    if cluster_feature_names is not None:
        cluster_features = np.array([getattr(cluster, feature) for feature in cluster_feature_names])
        cluster_features = (cluster_features - cluster_feature_means) / cluster_feature_stds
    else:
        cluster_features = None

    # Normalize member source features which are used in the support set
    members = (members[source_feature_names].to_numpy() - source_feature_means) / source_feature_stds

    for sample_idx in tqdm(range(n_samples), total=n_samples, desc='Evaluating candidate samples'):
        # Create a copy of the sources to be evaluated and replace the original properties with sampled properties
        candidates_sample = candidates.copy()
        candidates_sample[sample_properties] = samples[sample_idx]

        # Update the feature values for each sample
        add_features(candidates_sample, cluster, radius_feature=False, proper_motion_feature=update_f_pm,
                     parallax_feature=update_f_plx, isochrone_features=update_f_iso, fast_mode=fast_mode)

        # Normalization
        candidates_sample = (candidates_sample[source_feature_names].to_numpy() -
                             source_feature_means) / source_feature_stds

        # Create a normalized deep sets dataset for to-be-evaluated sources
        candidates_sample_dataset = deep_sets_eval_dataset(candidates_sample, members,
                                                           global_features=cluster_features,
                                                           size_support_set=size_support_set)

        # Predict membership based on the sampled properties
        candidates_sample_predictions[sample_idx] = predict_membership(candidates_sample_dataset, model)

    probabilities = np.mean(candidates_sample_predictions, axis=0)

    candidates['PMemb'] = probabilities
    candidates.to_csv(os.path.join(cluster_dir, 'candidates.csv'))
