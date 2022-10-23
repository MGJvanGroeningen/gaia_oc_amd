import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import csgraph_from_dense, minimum_spanning_tree

from gaia_oc_amd.data_preparation.features import radius_feature_function
from gaia_oc_amd.utils import projected_coordinates


def make_density_profile(members, cluster):
    """Makes a surface stellar density profile, consisting of a set of radius-density pairs,
    based on the cluster members.

    Args:
        members (Dataframe): Dataframe containing member sources
        cluster (Cluster): Cluster object

    Returns:
        rs (float, array): Radii of the density profile
        areas (float, array): Areas of the concentric shells of the density profile
        densities (float, array): Densities of the density profile
        sigmas (float, array): Density errors of the density profile

    """
    r = members.apply(radius_feature_function(cluster.ra, cluster.dec, cluster.dist), axis=1)
    r_max = r.max()

    # Bins that define the radius boundaries of the concentric shells in which to count members
    if r_max > 25:
        bins = np.concatenate((np.arange(10, step=0.5), np.logspace(1, np.log10(r_max), num=15, endpoint=True)))
    elif r_max < 3:
        bins = np.array([0, r_max / 2, r_max])
    else:
        bins = np.arange(int(r_max))

    rs = []
    densities = []
    sigmas = []
    areas = []

    # For each bin, determine: the mean radius of the contained members, the area of the concentric shell,
    # the density members in the shell and its uncertainty
    for i in range(len(bins) - 1):
        n_sources_in_bin = sum(np.where((bins[i] < r) & (r < bins[i + 1]), 1, 0))
        if n_sources_in_bin != 0:
            mean_r = sum(np.where((bins[i] < r) & (r < bins[i + 1]), r, 0)) / n_sources_in_bin
            area = np.pi * (bins[i + 1] ** 2 - bins[i] ** 2)
            density = n_sources_in_bin / area
            density_sigma = np.sqrt(n_sources_in_bin) / area

            rs.append(mean_r)
            areas.append(area)
            densities.append(density)
            sigmas.append(density_sigma)

    rs = np.array(rs)
    areas = np.array(areas)
    densities = np.array(densities)
    sigmas = np.array(sigmas)

    return rs, areas, densities, sigmas,


def king_model(rs, k, r_c, r_t, c):
    """Calculates the surface stellar density at a given radius from a cluster with central density k,
    core radius r_c, tidal radius r_t and background density c as defined by the King model.

    Args:
        rs (float, array): Array of radii for which to calculate the density
        k (float): Quantity related to the central density of the cluster
        r_c (float): Core radius
        r_t (float): Tidal radius
        c (float): Background density

    Returns:
        density (float, array): Surface stellar density

    """
    def before_plateau(r):
        r_rc = 1 / np.sqrt(1 + (r / r_c)**2)
        rt_rc = 1 / np.sqrt(1 + (r_t / r_c)**2)
        return k * (r_rc - rt_rc)**2 + c
    density = np.where(rs < r_t, before_plateau(rs), c)
    return density


def king_fit_log_likelihood(theta, r, density, area):
    """Determines the log likelihood of a set of King model parameters for a given distribution of the
    surface stellar density. The number of stars in a concentric shell around the cluster follows a
    Poisson distribution.

    Args:
        theta (float, tuple): King model parameters (k, r_c, r_t, c)
        r (float, array): Radii data
        density (float, array): Surface stellar density data
        area (float, array): The areas of the concentric shells with radius r

    Returns:
        log_likelihood (float, array): Log likelihood of the King model parameters.

    """
    k, r_c, r_t, c = theta
    king_density = king_model(r, k, r_c, r_t, c)
    f_x = king_density * area
    y = density * area
    log_likelihood = np.sum(np.log(f_x) * y - f_x - (y * np.log(y) - y + 1))
    return log_likelihood


def fit_king_model(r, density, area):
    """Fits a King model to surface stellar density distribution given by a set of radius-density pairs.

    Args:
        r (float, array): Radii data
        density (float, array): Surface stellar density data
        area (float, array): The areas of the concentric shells with radius r

    Returns:
        r_fit (float, array): Radii of the best fitting King model
        density_fit (float, array): Surface stellar densities of the best fitting King model
        best_r_t (float): Most likely tidal radius.

    """
    # Initial values and search space for finding the maximum log likelihood parameters.
    initial_values = np.array([density[0], 1, 30, density[-1]])
    bounds = [(1e-4, 200), (1e-4, 20), (1e-4, 200), (1e-4, 5.0)]

    # Minimize -log_likelihood
    soln = minimize(lambda *args: -king_fit_log_likelihood(*args), initial_values, args=(r, density, area),
                    bounds=bounds)
    best_k, best_r_c, best_r_t, best_c = soln.x

    r_fit = np.linspace(0, np.max(r) + 1)
    density_fit = king_model(r_fit, best_k, best_r_c, best_r_t, best_c)
    return r_fit, density_fit, best_r_t


def tidal_radius(members, cluster):
    """Calculates the tidal radius of the King model that best fits the members of a cluster.

    Args:
        members (Dataframe): Dataframe containing member sources
        cluster (Cluster): Cluster object

    Returns:
        r_t (float): Tidal radius of the cluster

    """
    rs, areas, densities, _ = make_density_profile(members, cluster)
    _, _, r_t = fit_king_model(rs, densities, areas)
    return r_t


def minimum_spanning_tree_length(x, y):
    """Calculates the length of the minimum spanning tree of a set of points.

    Args:
        x (float, array): x coordinates
        y (float, array): y coordinates

    Returns:
        min_spanning_tree_length (float): Length of the minimum spanning tree

    """
    pos = np.stack((x, y), axis=-1)
    dist_matrix = distance_matrix(pos, pos)
    graph = csgraph_from_dense(dist_matrix)
    min_spanning_tree_length = np.sum(minimum_spanning_tree(graph).toarray())
    return min_spanning_tree_length


def make_mass_segregation_profile(members, cluster, min_n_mst=5, max_n_mst=100, n_samples=20):
    """Makes a mass segregation profile, which defines the ratio of the minimum spanning tree length of
    the N most massive members and the mean minimum spanning tree length of a random set of N members.

    Args:
        members (Dataframe): Dataframe containing member sources
        cluster (Cluster): Cluster object
        min_n_mst (int): Minimum number of members for which to calculate the mass segregation ratio
        max_n_mst (int): Maximum number of members for which to calculate the mass segregation ratio
        n_samples (int): The number of samples of the random set of N members.

    Returns:
        n_mst (int, array): The number of members
        lambda_msr (float, array): The mass segregation ratio
        sigmas (float, array): The uncertainty of the mass segregation ratio

    """
    n_members = len(members)

    # Use the G magnitude to sort the members by mass
    members = members.sort_values(by=['phot_g_mean_mag']).copy()

    # Get the projected coordinates of the (sorted) members
    ra, dec = members['ra'], members['dec']
    x, y = projected_coordinates(ra, dec, cluster.ra, cluster.dec, cluster.dist)
    x, y = x.values, y.values

    n_mst = np.arange(min_n_mst, min(n_members, max_n_mst))
    lambda_msr = []
    sigmas = []

    for n in n_mst:
        # First determine the minimum spanning tree length of the N most massive members
        l_massive = minimum_spanning_tree_length(x[:n], y[:n])

        # Then do the same for a number of random samples of N members
        l_randoms = []
        for _ in range(n_samples):
            random_indices = np.random.choice(n_members, n, replace=False)
            l_random = minimum_spanning_tree_length(x[random_indices], y[random_indices])
            l_randoms.append(l_random)

        # Mean and standard deviation of samples
        lambda_msr.append(np.mean(np.array(l_randoms)) / l_massive)
        sigmas.append(np.std(np.array(l_randoms)) / l_massive)

    return n_mst, lambda_msr, sigmas
