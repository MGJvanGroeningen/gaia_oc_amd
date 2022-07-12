import numpy as np


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
                property_corr_label = properties[i] + '_' + properties[j] + '_corr'
                if property_corr_label in source_columns:
                    corr = sources[property_corr_label].to_numpy()
                    sigma_i = sources[property_errors[i]].to_numpy()
                    sigma_j = sources[property_errors[j]].to_numpy()
                    cov = corr * sigma_i * sigma_j
                else:
                    cov = 0
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
