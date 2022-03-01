import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import csgraph_from_dense, minimum_spanning_tree
from data_filters import projected_coordinates


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
    return np.sum(-model - (n * np.log(n) - n + 1) + np.log(model) * n)


def fit_king_model(x, y, area):
    k_i = y[0]
    r_c_i = 1
    r_t_i = 30
    c_i = y[-1]

    np.random.seed(42)
    initial = np.array([k_i, r_c_i, r_t_i, c_i])
    bounds = [(1e-4, 100), (1e-4, 20), (1e-4, 1000), (1e-3, 1)]
    soln = minimize(lambda *args: -king_fit_log_likelihood(*args), initial, args=(x, y, area), bounds=bounds)
    k_ml, r_c_ml, r_t_ml, c_ml = soln.x

    print(f'Cluster threshold radius = {r_t_ml} pc')
    theta = (k_ml, r_c_ml, r_t_ml, c_ml)
    print(f'Log likelihood: {king_fit_log_likelihood(theta, x, y, area)}')

    x_fit = np.linspace(0, np.max(x) + 1)
    y_fit = king_model(x_fit, k_ml, r_c_ml, r_t_ml, c_ml)
    return x_fit, y_fit, r_t_ml


def make_density_profile(members, cluster_kwargs=None):
    x, y = projected_coordinates(members, cluster_kwargs)

    r = np.sqrt(x ** 2 + y ** 2)
    r_max = np.max(r)

    n_bins = 16
    if r_max > 20:
        bins = np.concatenate((np.arange(10), np.logspace(1, np.log10(r_max), num=n_bins, endpoint=True)))
    else:
        bins = np.arange(int(r_max))
    # bins = np.linspace(0, 80, 80)

    rs = []
    densities = []
    sigmas = []
    areas = []

    for i in range(len(bins) - 1):
        n_sources_in_bin = sum(np.where((bins[i] < r) & (r < bins[i + 1]), 1, 0))
        mean_r = sum(np.where((bins[i] < r) & (r < bins[i + 1]), r, 0)) / n_sources_in_bin
        area = np.pi * (bins[i + 1] ** 2 - bins[i] ** 2)
        density = n_sources_in_bin / area
        density_sigma = np.sqrt(n_sources_in_bin) / area

        rs.append(mean_r)
        densities.append(density)
        sigmas.append(density_sigma)
        areas.append(area)

    return np.array(rs), np.array(densities), np.array(sigmas), np.array(areas)


def make_mass_segregation_profile(members, cluster_kwargs=None):
    n_members = len(members)
    members = members.sort_values(by=['phot_g_mean_mag']).copy()
    x, y = projected_coordinates(members, cluster_kwargs)
    # plt.scatter(x, y, c=members['phot_g_mean_mag'], s=4.0)
    # plt.colorbar()
    # plt.show()

    min_n_mst = 5
    max_n_mst = min(100, n_members)
    n_samples = 20

    bins = np.arange(min_n_mst, max_n_mst)

    ms = []
    sigmas = []

    for n in np.arange(min_n_mst, max_n_mst):
        massive_points = np.concatenate((x[:n][..., None], y[:n][..., None]), axis=1)
        # plt.scatter(massive_points[:, 0], massive_points[:, 1])
        # plt.title('massive')
        # plt.show()
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
