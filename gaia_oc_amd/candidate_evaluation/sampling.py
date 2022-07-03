import numpy as np
from scipy.stats import multivariate_normal


def get_corr_idx(row, col):
    return 3 * min(row, col) + max(row, col) - int(np.sum(np.arange(min(row, col) + 2)))


def mean_and_cov_matrix(sources):
    n_sources = len(sources)

    sample_properties = ['parallax', 'pmra', 'pmdec', 'phot_g_mean_mag', 'bp_rp']
    sample_error_properties = ['parallax_error', 'pmra_error', 'pmdec_error', 'phot_g_mean_mag_error', 'bp_rp_error']
    astrometric_corr_properties = ['parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr']

    n_astrometric_properties = 3
    n_properties = len(sample_properties)

    means = np.zeros((n_properties, n_sources))
    cov_matrices = np.zeros((n_properties, n_properties, n_sources))

    for i in range(n_properties):
        if i < n_astrometric_properties:
            means[i] = sources[sample_properties[i]]
            for j in range(i, n_astrometric_properties):
                if i == j:
                    sigma = sources[sample_error_properties[i]].to_numpy()
                    cov_matrices[i, j] = sigma ** 2
                else:
                    corr_name_idx = get_corr_idx(i, j)
                    corr = sources[astrometric_corr_properties[corr_name_idx]].to_numpy()
                    sigma_i = sources[sample_error_properties[i]].to_numpy()
                    sigma_j = sources[sample_error_properties[j]].to_numpy()
                    cov = corr * sigma_i * sigma_j
                    cov_matrices[i, j] = cov
                    cov_matrices[j, i] = cov
        else:
            means[i] = sources[sample_properties[i]]
            sigma = sources[sample_error_properties[i]].to_numpy()
            cov_matrices[i, i] = sigma ** 2
    return means, cov_matrices


def sample_sources(sources, means, cov_matrices):
    sources_sample = sources.copy()
    n_sources = len(sources_sample)

    sample_properties = ['parallax', 'pmra', 'pmdec', 'phot_g_mean_mag', 'bp_rp']
    n_properties = len(sample_properties)

    samples = np.zeros((n_sources, n_properties))
    for i in range(n_sources):
        samples[i] = multivariate_normal(mean=means[:, i], cov=cov_matrices[:, :, i]).rvs(1)

    sources_sample[sample_properties] = samples

    return sources_sample
