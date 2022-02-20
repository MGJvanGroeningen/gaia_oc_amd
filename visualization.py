import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
from scipy.special import factorial
from scipy.sparse.csgraph import csgraph_from_dense, minimum_spanning_tree
from filter_cone_data import most_likely_value, ra_dec_to_xy


def king_model(r, k, r_c, r_t, c):
    def before_plateau(r_):
        r_rc = 1 / np.sqrt(1 + (r_ / r_c)**2)
        rt_rc = 1 / np.sqrt(1 + (r_t / r_c)**2)
        return k * (r_rc - rt_rc)**2 + c
    n = np.where(r < r_t, before_plateau(r), c)
    return n


def fit_king_model(x, y):
    def log_likelihood(theta, r, y):
        k, r_c, r_t, c = theta
        model = king_model(r, k, r_c, r_t, c)
        return np.sum(-model - np.log(factorial(y)) + np.log(model) * y)

    k_i = y[0]
    r_c_i = 1
    r_t_i = 30
    c_i = y[-1]

    np.random.seed(42)
    initial = np.array([k_i, r_c_i, r_t_i, c_i])
    bounds = [(1e-4, 100), (1e-4, 100), (1e-4, 200), (1e-4, 10)]
    soln = minimize(lambda *args: -log_likelihood(*args), initial, args=(x, y), bounds=bounds)
    k_ml, r_c_ml, r_t_ml, c_ml = soln.x

    print(f'Cluster threshold radius = {r_t_ml} pc')

    x_fit = np.linspace(0, np.max(x) + 1)
    y_fit = king_model(x_fit, k_ml, r_c_ml, r_t_ml, c_ml)
    return x_fit, y_fit, r_t_ml


def save_csv(cluster_name, save_dir, members_df, noise_df, member_candidates_df, non_member_candidates_df, run_suffix=None,
             combined=True):
    if not os.path.exists(os.path.join('results', cluster_name)):
        os.mkdir(os.path.join(save_dir, 'results', cluster_name))

    if run_suffix is not None:
        suffix = '_' + run_suffix
    else:
        suffix = ''

    sources = [members_df, noise_df, member_candidates_df, non_member_candidates_df]
    ids = [0, 1, 2, 3]
    source_labels = ['members', 'noise', 'member_candidates', 'non_member_candidates']

    if combined:
        for source, idx in zip(sources, ids):
            source['id'] = idx

        cone_df = pd.concat(sources, sort=False, ignore_index=True)
        cone_df.to_csv(os.path.join(save_dir, 'results', cluster_name, 'cone' + suffix + '.csv'))
    else:
        for source, label in zip(sources, source_labels):
            source.to_csv(os.path.join(save_dir, 'results', cluster_name, f'{label}{suffix}.csv'))


def load_csv(cluster_name, save_dir, run_suffix=None):
    if run_suffix is not None:
        suffix = '_' + run_suffix
    else:
        suffix = ''

    csv_file = os.path.join(save_dir, 'results', cluster_name, f'cone{suffix}.csv')
    cone_df = pd.read_csv(csv_file)

    members = cone_df[cone_df['id'] == 0].copy()
    noise = cone_df[cone_df['id'] == 1].copy()
    member_candidates = cone_df[cone_df['id'] == 2].copy()
    non_member_candidates = cone_df[cone_df['id'] == 3].copy()

    candidates = pd.concat((member_candidates, non_member_candidates), axis=0)
    return members, noise, candidates, member_candidates, non_member_candidates


def make_plots(cluster_name, save_dir, members_df, prob_threshold, noise_df=None, candidates_df=None,
               member_candidates_df=None, non_member_candidates_df=None, isochrone_df=None, mean_predictions=None,
               zoom=1.0, dot_size=1.0, alpha=0.8, cmap='veridis', run_suffix=None, plot=True, save=True):

    if candidates_df is not None:
        suffix = 'candidates'
    else:
        suffix = 'members'

    if run_suffix is not None:
        suffix2 = '_' + run_suffix
    else:
        suffix2 = ''

    plot_kwargs = [{'x': 'ra',
                    'y': 'dec'},
                   {'x': 'phot_g_mean_mag',
                    'y': 'parallax'},
                   {'x': 'pmra',
                    'y': 'pmdec'},
                   {'x': 'bp_rp',
                    'y': 'phot_g_mean_mag'}]

    if not os.path.exists(f'results/{cluster_name}'):
        os.mkdir(os.path.join(save_dir, 'results', cluster_name))

    plot_ranges = {}
    for field in ['phot_g_mean_mag', 'parallax', 'pmra', 'pmdec', 'bp_rp']:
        if member_candidates_df is not None and len(member_candidates_df) > 0:
            field_min = min(member_candidates_df[field].min(), members_df[field].min())
            field_max = max(member_candidates_df[field].max(), members_df[field].min())
        elif non_member_candidates_df is not None:
            field_min = min(non_member_candidates_df[field].min(), members_df[field].min())
            field_max = max(non_member_candidates_df[field].max(), members_df[field].min())
        elif candidates_df is not None:
            field_min = candidates_df[field].min()
            field_max = candidates_df[field].max()
        else:
            field_min = members_df[field].min()
            field_max = members_df[field].max()
        field_range = field_max - field_min
        center = field_min + field_range / 2
        plot_ranges.update({field: [center - field_range / zoom, center + field_range / zoom]})

    if mean_predictions is not None:
        fig_size = (10, 8)
    else:
        fig_size = (8, 8)

    for dic in plot_kwargs:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)

        if noise_df is not None:
            ax.scatter(noise_df[dic['x']], noise_df[dic['y']],
                       label='non members', s=0.2 * dot_size, c='gray', alpha=0.2, rasterized=True)
        if candidates_df is not None:
            if mean_predictions is not None:
                candidate_color = mean_predictions
            else:
                candidate_color = 'orange'
            colored_ax = ax.scatter(candidates_df[dic['x']], candidates_df[dic['y']],
                                    label=f'candidates', s=dot_size, c=candidate_color, rasterized=True, cmap=cmap)
            if mean_predictions is not None:
                c_bar = fig.colorbar(colored_ax, ax=ax)
                c_bar.set_label("Membership probability", fontsize=16)
        if member_candidates_df is not None:
            ax.scatter(member_candidates_df[dic['x']], member_candidates_df[dic['y']], marker='+',
                       label=f'candidate = member (>{prob_threshold})', s=3 * dot_size, c='orange', rasterized=True)
        if non_member_candidates_df is not None:
            ax.scatter(non_member_candidates_df[dic['x']], non_member_candidates_df[dic['y']],
                       label=f'candidate = non member', s=dot_size, c='red', rasterized=True)
        ax.scatter(members_df[dic['x']], members_df[dic['y']],
                   label=f'members (>{prob_threshold})', s=dot_size, c='blue', alpha=alpha, rasterized=True)

        if dic['x'] in plot_ranges:
            ax.set_xlim(plot_ranges[dic['x']][0], plot_ranges[dic['x']][1])
        if dic['y'] in plot_ranges:
            ax.set_ylim(plot_ranges[dic['y']][0], plot_ranges[dic['y']][1])
        if dic['y'] == 'phot_g_mean_mag':
            if isochrone_df is not None:
                ax.plot(isochrone_df['bp_rp'], isochrone_df['phot_g_mean_mag'], label='isochrone')
            ax.invert_yaxis()
        ax.set_title(f'{cluster_name}', fontsize=18)
        ax.set_xlabel(dic['x'], fontsize=16)
        ax.set_ylabel(dic['y'], fontsize=16)
        ax.legend(fontsize='small')

        if save:
            plt.savefig(os.path.join(save_dir, 'results', cluster_name, f"{dic['x']}_{dic['y']}_{suffix}{suffix2}.png"))
        if plot:
            plt.show()
        plt.close(fig)


def print_sets(member_candidates, lp_members, noise, candidates):
    # create various sets of source identities belonging to different groups
    candidate_ids = set(candidates['source_id'].to_numpy())
    noise_ids = set(noise['source_id'].to_numpy())
    lp_member_ids = set(lp_members['source_id'].to_numpy())
    member_candidates_ids = set(member_candidates['source_id'].to_numpy())

    # mean_predicted_member_prob = np.mean(mean_predictions[member_candidates_indices])
    # mean_predicted_non_member_prob = np.mean(np.delete(mean_predictions, member_candidates_indices))

    # check how many of the database members which were not used for training
    # are among the sources selected as candidates or noise
    # and whether they were identified by the model as member or as non member
    for ids, name in zip([candidate_ids, noise_ids], ['candidate', 'noise']):
        n = len(ids)
        n_lp_member = len(ids & lp_member_ids)
        n_lp_member_member = len(ids & lp_member_ids & member_candidates_ids)
        n_lp_member_not_member = len(ids & lp_member_ids - member_candidates_ids)
        n_not_lp_member = len(ids - lp_member_ids)
        n_not_lp_member_member = len(ids - lp_member_ids & member_candidates_ids)
        n_not_lp_member_not_member = len(ids - lp_member_ids - member_candidates_ids)

        print(' ')
        print(f'{name} total', n)
        print(f'     {name} = lp member', n_lp_member)
        if name == 'candidate':
            print(f'         {name} = lp member & member', n_lp_member_member)
            print(f'         {name} = lp member & not member', n_lp_member_not_member)
        print(f'     {name} = not lp member', n_not_lp_member)
        if name == 'candidate':
            print(f'         {name} = not lp member & member', n_not_lp_member_member)
            print(f'         {name} = not lp member & not member', n_not_lp_member_not_member)


def projected_coordinates(members, cluster_kwargs):
    rad_per_deg = np.pi / 180

    ra = members['ra'] * rad_per_deg
    dec = members['dec'] * rad_per_deg

    if cluster_kwargs is not None:
        d = cluster_kwargs['dist']
        ra_c = cluster_kwargs['ra'] * rad_per_deg
        dec_c = cluster_kwargs['dec'] * rad_per_deg
    else:
        d = 1000 / (most_likely_value(members['parallax'], members['parallax_error']) + 0.029)
        ra_c = most_likely_value(members['ra'], members['ra_error']) * rad_per_deg
        dec_c = most_likely_value(members['dec'], members['dec_error']) * rad_per_deg

    x, y = ra_dec_to_xy(ra, dec, ra_c, dec_c, d)
    return x.values, y.values


def make_density_profile(members, cluster_kwargs=None):
    x, y = projected_coordinates(members, cluster_kwargs)

    r = np.sqrt(x ** 2 + y ** 2)
    r_max = np.max(r)

    n_bins = 12
    if r_max > 20:
        bins = np.concatenate((np.arange(10), np.logspace(1, np.log10(r_max), num=n_bins, endpoint=True)))
    else:
        bins = np.arange(int(r_max))

    rs = []
    densities = []
    sigmas = []

    for i in range(len(bins) - 1):
        n_sources_in_bin = sum(np.where((bins[i] < r) & (r < bins[i + 1]), 1, 0))
        mean_r = sum(np.where((bins[i] < r) & (r < bins[i + 1]), r, 0)) / n_sources_in_bin
        area = np.pi * (bins[i + 1] ** 2 - bins[i] ** 2)
        density = n_sources_in_bin / area
        density_sigma = np.sqrt(n_sources_in_bin) / area

        rs.append(mean_r)
        densities.append(density)
        sigmas.append(density_sigma)

    return np.array(rs), np.array(densities), np.array(sigmas)


def plot_density_profile(cluster_name, save_dir, members, prob_threshold, cluster_kwargs=None, run_suffix=None,
                         plot=True, save=True):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if run_suffix is not None:
        suffix = '_' + run_suffix
    else:
        suffix = ''

    if type(members) == list:
        rs, densities, sigmas = make_density_profile(members[0], cluster_kwargs)
        ax.errorbar(rs, densities, sigmas, fmt="o", capsize=3.0, label='old members')
        rs, densities, sigmas = make_density_profile(members[1], cluster_kwargs)
        x, f_x, r_t = fit_king_model(rs, densities)
        ax.plot(x, f_x, label=r'king model fit ($R_{t}$' + f'={np.round(r_t, 1)} pc)')
        ax.errorbar(rs, densities, sigmas, fmt="o", capsize=3.0, label='old + new members')
        ax.legend()
    else:
        rs, densities, sigmas = make_density_profile(members, cluster_kwargs)
        x, f_x, r_t = fit_king_model(rs, densities)
        ax.plot(x, f_x, label='ml_fit')
        ax.errorbar(rs, densities, sigmas, fmt="o")

    ax.set_yscale('log')
    ax.set_title(f'Density profile of {cluster_name} with >{prob_threshold} members', fontsize=16)
    ax.set_xlabel('r [pc]', fontsize=14)
    ax.set_ylabel(r'$\rho$ [stars / pc$^{2}$]', fontsize=14)
    if save:
        plt.savefig(os.path.join(save_dir, 'results', cluster_name, f"density_profile{suffix}.png"))
    if plot:
        plt.show()
    plt.close(fig)
    return r_t


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


def plot_mass_segregation_profile(cluster_name, save_dir, members, prob_threshold, cluster_kwargs=None,
                                  run_suffix=None, plot=True, save=True):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if run_suffix is not None:
        suffix = '_' + run_suffix
    else:
        suffix = ''

    if type(members) == list:
        bins, ms, sigmas = make_mass_segregation_profile(members[0], cluster_kwargs)
        ax.errorbar(bins, ms, sigmas, elinewidth=1.0, ms=2.0, capsize=2.0, fmt="o", c='gray', label='old members')
        bins, ms, sigmas = make_mass_segregation_profile(members[1], cluster_kwargs)
        ax.errorbar(bins, ms, sigmas, elinewidth=0.5, ms=2.0, capsize=2.0, fmt="o", c='black',
                    label='old + new members')
        ax.legend()
    else:
        bins, ms, sigmas = make_mass_segregation_profile(members, cluster_kwargs)
        ax.errorbar(bins, ms, sigmas, elinewidth=1.0, ms=2.0, capsize=2.0, fmt="o", c='black')

    ax.hlines(1, np.min(bins), np.max(bins), linestyles='dashed')
    ax.set_title(f'Mass segregation of {cluster_name} with >{prob_threshold} members', fontsize=16)
    ax.set_xlabel(r'$N_{MST}$', fontsize=14)
    ax.set_ylabel(r'$\Lambda_{MSR}$', fontsize=14)
    if save:
        plt.savefig(os.path.join(save_dir, 'results', cluster_name, f"mass_segregation{suffix}.png"))
    if plot:
        plt.show()
    plt.close(fig)


def compare_members(cluster_name, save_dir, old_members, new_members, members_to_compare_to, member_prob_threshold,
                    x_fields, y_fields, run_suffix=None, plot=True, save=True):
    n_old = len(old_members)
    n_new = len(new_members)
    n_compare = len(members_to_compare_to)

    old_members_ids = set(old_members['source_id'].to_numpy())
    new_members_ids = set(new_members['source_id'].to_numpy())
    compare_members_ids = set(members_to_compare_to['source_id'].to_numpy())

    old_duplicates = len(old_members_ids & compare_members_ids)
    new_duplicates = len(new_members_ids & compare_members_ids)

    old_plus_new_members_ids = old_members_ids.copy()
    old_plus_new_members_ids.update(new_members_ids)
    compare_duplicates = len(old_plus_new_members_ids & compare_members_ids)

    old_only = n_old - old_duplicates
    new_only = n_new - new_duplicates
    compare_only = n_compare - compare_duplicates

    print(' ')
    print('Sources in the old set:', n_old)
    print('Sources only in set 1:', old_only)

    print(' ')
    print('Sources in the new set:', n_new)
    print('Sources only in set 2:', new_only)

    print(' ')
    print('Sources in the compare set:', n_compare)
    print('Sources only in the compare set:', compare_only)

    print(' ')
    print('Sources in old & compare:', old_duplicates)
    print('Sources in new & compare:', new_duplicates)
    print('Sources in old or new & compare:', compare_duplicates)

    if run_suffix is not None:
        suffix = '_' + run_suffix
    else:
        suffix = ''

    for x, y in zip(x_fields, y_fields):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        ax.scatter(new_members[x], new_members[y], label=f'new members (>{member_prob_threshold})', s=16.0, c='red')
        ax.scatter(old_members[x], old_members[y], label=f'old members (>{member_prob_threshold})', s=8.0, c='blue')
        ax.scatter(members_to_compare_to[x], members_to_compare_to[y],
                   label=f'compare members (>{member_prob_threshold})', s=4.0, c='green')
        ax.set_title(f'{cluster_name} comparison between member sets', fontsize=16)
        ax.set_xlabel(x, fontsize=14)
        ax.set_ylabel(y, fontsize=14)
        if y == 'phot_g_mean_mag':
            ax.invert_yaxis()
        ax.legend()
        if save:
            plt.savefig(os.path.join(save_dir, 'results', cluster_name, f"member_comparison_{x}_{y}{suffix}.png"))
        if plot:
            plt.show()
        plt.close(fig)
