import numpy as np
from tqdm import tqdm
from torch.nn.functional import softmax

from gaia_oc_amd.candidate_evaluation.sampling import sample_sources
from gaia_oc_amd.data_preparation.datasets import deep_sets_eval_dataset
from gaia_oc_amd.data_preparation.utils import add_columns
from gaia_oc_amd.data_preparation.features import Features


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


def calculate_probabilities(eval_sources, model, cluster, isochrone, members, training_features, training_feature_means,
                            training_feature_stds, size_support_set=5, n_samples=40, seed=42):
    """Returns the membership probabilities of a set of sources, based on the model predictions
    for a number of sampled properties of these sources.

    Args:
        eval_sources (Dataframe): Dataframe containing sources for which we want to determine
            the membership probability
        model (nn.Module): Deep sets model
        cluster (Cluster): Cluster object
        isochrone (Dataframe): Dataframe containing colour and magnitude values of the isochrone curve.
        members (Dataframe): Dataframe containing member sources (used for the support sets)
        training_features (str, list): Labels of the training features
        training_feature_means (float, array): Mean values of the training features (normalization)
        training_feature_stds (float, array): Standard deviation of the training features (normalization)
        size_support_set (int): Number of members in the support set
        n_samples (int): The number of samples
        seed (int): The random seed, which determines the samples

    Returns:
        probabilities (float, array): Array of membership probabilities for the to-be-evaluated sources

    """
    model.eval()

    # Sample astrometric and photometric properties of the sources
    sample_properties = ['parallax', 'pmra', 'pmdec', 'phot_g_mean_mag', 'bp_rp']
    samples = sample_sources(eval_sources, sample_properties=sample_properties, n_samples=n_samples)

    # We may need to re-calculate some of the training feature values of the sources,
    # if they depend on the sampled properties
    custom_features = Features(cluster, isochrone)
    features_to_be_updated = []
    for feature in custom_features.update_after_sample_features:
        if feature.label in training_features:
            features_to_be_updated.append(feature)

    np.random.seed(seed)
    eval_sources_sample_predictions = np.zeros((n_samples, len(eval_sources)))

    for sample_idx in tqdm(range(n_samples), total=n_samples, desc='Evaluating candidate samples'):
        eval_sources_sample = eval_sources.copy()
        eval_sources_sample[sample_properties] = samples[sample_idx]

        # After sampling the properties, we update the feature values
        add_columns([eval_sources_sample], [feature.function for feature in features_to_be_updated],
                    [feature.label for feature in features_to_be_updated])

        # Create a normalized deep sets dataset for to-be-evaluated sources
        eval_sources_sample_dataset = deep_sets_eval_dataset(eval_sources_sample, members,
                                                             training_features, training_feature_means,
                                                             training_feature_stds, size_support_set)

        eval_sources_sample_predictions[sample_idx] = predict_membership(eval_sources_sample_dataset, model)

    probabilities = np.mean(eval_sources_sample_predictions, axis=0)

    return probabilities
