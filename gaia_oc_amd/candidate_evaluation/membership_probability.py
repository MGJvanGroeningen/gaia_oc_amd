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

    means = np.zeros((n_properties, n_sources), dtype=np.float32)
    cov_matrices = np.zeros((n_properties, n_properties, n_sources), dtype=np.float32)

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
    means = np.swapaxes(means, 0, 1)
    cov_matrices = np.swapaxes(cov_matrices, 0, 2)
    return means, cov_matrices


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


def calculate_candidate_probs(cluster_dir, model_dir, n_samples=40, size_support_set=10, hard_size_ss=True, seed=42):
    """Calculates the membership probabilities of a set of sources, based on the model predictions
    for a number of sampled properties of these sources. The probabilities are saved in the candidate csv file
    of the corresponding cluster.

    Args:
        cluster_dir (str): Directory where the cluster data is stored
        model_dir (str): Directory where the model data is stored
        n_samples (int): The number of samples
        size_support_set (int): The number of members in the support set. Ideally, this is the same as the
            support set size used in the training set, however results depend less on the size of the support set
            when using larger numbers of samples. For clusters with less training members than the support set size,
            the number of available training members is used for the support set size instead.
        hard_size_ss (bool): When false, set the support set size to the number of available training members when the
            former is larger than the latter.
        seed (int): The random seed, which determines the samples

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

    # Set the support set size to the number of available training members when the latter is smaller
    if not hard_size_ss:
        size_support_set = min(size_support_set, len(members))

    # Sample astrometric and photometric properties of the sources
    np.random.seed(seed)
    sample_properties = ['parallax', 'pmra', 'pmdec', 'phot_g_mean_mag', cluster.isochrone_colour]

    # Determine the means and covariance matrix of the astrometric and photometric properties of the sources
    property_means, property_cov_matrices = means_and_covariance_matrix(candidates, sample_properties)
    n_sources = len(candidates)
    n_properties = len(sample_properties)

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

    samples_shape = (n_sources, n_properties)
    cholesky_matrix = np.linalg.cholesky(property_cov_matrices)

    for sample_idx in tqdm(range(n_samples), total=n_samples, desc='Evaluating candidate samples'):
        # Create a copy of the sources to be evaluated and replace the original properties with sampled properties
        candidates_sample = candidates.copy()
        normal = np.random.standard_normal((*samples_shape, 1))
        candidates_sample[sample_properties] = (cholesky_matrix @ normal).reshape(samples_shape) + property_means

        # Update the feature values for each sample
        add_features(candidates_sample, cluster, radius_feature=False, proper_motion_feature=update_f_pm,
                     parallax_feature=update_f_plx, isochrone_features=update_f_iso)

        # Normalization
        candidates_sample = (candidates_sample[source_feature_names].to_numpy() -
                             source_feature_means) / source_feature_stds

        # Create a normalized deep sets dataset for to-be-evaluated sources
        candidates_sample_dataset = deep_sets_eval_dataset(candidates_sample, members,
                                                           global_features=cluster_features,
                                                           size_support_set=size_support_set)

        # Predict membership based on the sampled properties
        candidates_sample_predictions[sample_idx] = predict_membership(candidates_sample_dataset, model)

    # Take the mean of the samples
    probabilities = np.mean(candidates_sample_predictions, axis=0)

    # Save the membership probabilities
    candidates['PMemb'] = probabilities
    candidates.to_csv(os.path.join(cluster_dir, 'candidates.csv'))
