import numpy as np
from scipy.stats import multivariate_normal


def get_corr_idx(row, col):
    return 3 * min(row, col) + max(row, col) - int(np.sum(np.arange(min(row, col) + 2)))


def candidate_distribution(candidates):
    n_sources = len(candidates)

    sample_properties = ['parallax', 'pmra', 'pmdec', 'phot_g_mean_mag', 'bp_rp']
    sample_error_properties = ['parallax_error', 'pmra_error', 'pmdec_error', 'phot_g_mean_mag_error', 'bp_rp_error']
    astrometric_corr_properties = ['parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr']

    n_astrometric_properties = 3
    n_properties = len(sample_properties)

    means = np.zeros((n_properties, n_sources))
    cov_matrices = np.zeros((n_properties, n_properties, n_sources))

    for i in range(n_properties):
        if i < n_astrometric_properties:
            means[i] = candidates[sample_properties[i]]
            for j in range(i, n_astrometric_properties):
                if i == j:
                    sigma = candidates[sample_error_properties[i]].to_numpy()
                    cov_matrices[i, j] = sigma ** 2
                else:
                    corr_name_idx = get_corr_idx(i, j)
                    corr = candidates[astrometric_corr_properties[corr_name_idx]].to_numpy()
                    sigma_i = candidates[sample_error_properties[i]].to_numpy()
                    sigma_j = candidates[sample_error_properties[j]].to_numpy()
                    cov = corr * sigma_i * sigma_j
                    cov_matrices[i, j] = cov
                    cov_matrices[j, i] = cov
        else:
            means[i] = candidates[sample_properties[i]]
            sigma = candidates[sample_error_properties[i]].to_numpy()
            cov_matrices[i, i] = sigma ** 2
    return means, cov_matrices


def sample_candidates(candidates, means, cov_matrices):
    candidate_sample = candidates.copy()
    n_sources = len(candidate_sample)

    sample_properties = ['parallax', 'pmra', 'pmdec', 'phot_g_mean_mag', 'bp_rp']
    n_properties = len(sample_properties)

    samples = np.zeros((n_sources, n_properties))
    for i in range(n_sources):
        samples[i] = multivariate_normal(mean=means[:, i], cov=cov_matrices[:, :, i]).rvs(1)

    candidate_sample[sample_properties] = samples

    return candidate_sample
