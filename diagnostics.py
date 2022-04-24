import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import csgraph_from_dense, minimum_spanning_tree


def most_likely_value(values, errors):
    return np.sum(values / errors ** 2) / np.sum(1 / errors ** 2)


def projected_coordinates(members, cluster):
    rad_per_deg = np.pi / 180

    ra = members['ra'] * rad_per_deg
    dec = members['dec'] * rad_per_deg

    if cluster is not None:
        d = cluster.dist
        ra_c = cluster.ra * rad_per_deg
        dec_c = cluster.dec * rad_per_deg
    else:
        d = 1000 / (most_likely_value(members['parallax'], members['parallax_error']) + 0.029)
        ra_c = most_likely_value(members['ra'], members['ra_error']) * rad_per_deg
        dec_c = most_likely_value(members['dec'], members['dec_error']) * rad_per_deg

    x = d * np.sin(ra - ra_c) * np.cos(dec)
    y = d * (np.cos(dec_c) * np.sin(dec) - np.sin(dec_c) * np.cos(dec) * np.cos(ra - ra_c))
    return x, y


def king_model(r, k, r_c, r_t, c):
    def before_plateau(r_):
        r_rc = 1 / np.sqrt(1 + (r_ / r_c)**2)
        rt_rc = 1 / np.sqrt(1 + (r_t / r_c)**2)
        return k * (r_rc - rt_rc)**2 + c
    n = np.where(r < r_t, before_plateau(r), c)
    return n


def king_fit_log_likelihood(theta, r, y, area):
    k, r_c, r_t, c = theta
    model = king_model(r, k, r_c, r_t, c) * area
    n = y * area
    L = np.sum(np.log(model) * n - model - (n * np.log(n) - n + 1))
    return L


def king_fit_log_prior(theta):
    k, r_c, r_t, c = theta
    if 1.0 < k < 200.0 and 0.1 < r_c < 20.0 and 1.0 < r_t < 200.0 and 1e-4 < c < 5.0:
        return 0.0
    return -np.inf


def king_fit_log_probability(theta, x, y, area):
    lp = king_fit_log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + king_fit_log_likelihood(theta, x, y, area)


def fit_king_model(x, y, area):
    k_i = y[0]
    r_c_i = 1
    r_t_i = 30
    c_i = y[-1]

    np.random.seed(42)
    initial = np.array([k_i, r_c_i, r_t_i, c_i])
    bounds = [(1e-4, 200), (1e-4, 20), (1e-4, 200), (1e-4, 5.0)]
    soln = minimize(lambda *args: -king_fit_log_likelihood(*args), initial, args=(x, y, area), bounds=bounds)
    k_ml, r_c_ml, r_t_ml, c_ml = soln.x

    x_fit = np.linspace(0, np.max(x) + 1)
    y_fit = king_model(x_fit, k_ml, r_c_ml, r_t_ml, c_ml)
    return x_fit, y_fit, r_t_ml


def fit_king_model_mcmc(x, y, area):
    k_i = y[0]
    r_c_i = 1
    r_t_i = 30
    c_i = y[-1]

    np.random.seed(42)
    initial = np.array([k_i, r_c_i, r_t_i, c_i])
    bounds = [(1.0, 200), (0.1, 20), (1.0, 200), (1e-4, 5)]
    soln = minimize(lambda *args: -king_fit_log_likelihood(*args), initial, args=(x, y, area), bounds=bounds)

    pos = soln.x + np.array([0.1, 1.0, 1.0, 1e-4]) * np.random.uniform(size=(8, 4))
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, king_fit_log_probability, args=(x, y, area))
    sampler.run_mcmc(pos, 10000, progress=True)

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["k", "r_c", "r_t", "c"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])

    axes[-1].set_xlabel("step number")

    flat_samples = sampler.get_chain(discard=2000, thin=15, flat=True)

    corner.corner(flat_samples, labels=labels)

    k_ml, r_c_ml, r_t_ml, c_ml = soln.x

    x_fit = np.linspace(0, np.max(x) + 1)
    y_fit = king_model(x_fit, k_ml, r_c_ml, r_t_ml, c_ml)
    return x_fit, y_fit, r_t_ml


def approximate_tidal_radius(r_fit, density_fit, threshold=0.01):
    r_t = r_fit[-1]
    for i in np.flip(np.arange(len(r_fit) - 1)):
        dlogrho_dx = np.abs((np.log(density_fit[i + 1]) - np.log(density_fit[i])) / (r_fit[i + 1] - r_fit[i]))
        if dlogrho_dx > threshold:
            r_t = r_fit[i + 1]
            break
    return r_t


def make_density_profile(members, cluster):
    x, y = projected_coordinates(members, cluster)
    r = np.sqrt(x.values ** 2 + y.values ** 2)
    r_max = np.max(r)

    n_bins = 16
    if r_max > 20:
        bins = np.concatenate((np.arange(10, step=0.5), np.logspace(1, np.log10(r_max), num=n_bins, endpoint=True)))
    else:
        bins = np.arange(int(r_max))

    rs = []
    densities = []
    sigmas = []
    areas = []

    for i in range(len(bins) - 1):
        n_sources_in_bin = sum(np.where((bins[i] < r) & (r < bins[i + 1]), 1, 0))
        if n_sources_in_bin != 0:
            mean_r = sum(np.where((bins[i] < r) & (r < bins[i + 1]), r, 0)) / n_sources_in_bin
            area = np.pi * (bins[i + 1] ** 2 - bins[i] ** 2)
            density = n_sources_in_bin / area
            density_sigma = np.sqrt(n_sources_in_bin) / area

            rs.append(mean_r)
            densities.append(density)
            sigmas.append(density_sigma)
            areas.append(area)

    return np.array(rs), np.array(densities), np.array(sigmas), np.array(areas)


def make_mass_segregation_profile(members, cluster):
    n_members = len(members)
    members = members.sort_values(by=['phot_g_mean_mag']).copy()
    x, y = projected_coordinates(members, cluster)
    x, y = x.values, y.values

    min_n_mst = 5
    max_n_mst = min(100, n_members)
    n_samples = 20

    bins = np.arange(min_n_mst, max_n_mst)

    ms = []
    sigmas = []

    for n in np.arange(min_n_mst, max_n_mst):
        massive_points = np.concatenate((x[:n][..., None], y[:n][..., None]), axis=1)
        massive_dist_matrix = distance_matrix(massive_points, massive_points)
        massive_graph = csgraph_from_dense(massive_dist_matrix)
        l_massive = np.sum(minimum_spanning_tree(massive_graph).toarray())

        l_randoms = []
        for _ in range(n_samples):
            indices = np.random.choice(n_members, n, replace=False)
            random_points = np.concatenate((x[indices][..., None], y[indices][..., None]), axis=1)
            random_dist_matrix = distance_matrix(random_points, random_points)
            random_graph = csgraph_from_dense(random_dist_matrix)
            l_random = np.sum(minimum_spanning_tree(random_graph).toarray())
            l_randoms.append(l_random)

        ms.append(np.mean(np.array(l_randoms)) / l_massive)
        sigmas.append(np.std(np.array(l_randoms)) / l_massive)

    return bins, ms, sigmas
