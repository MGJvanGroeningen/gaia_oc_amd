import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diagnostics import make_density_profile, make_mass_segregation_profile, fit_king_model
from data_filters import crop_member_candidates


def plot_sources(cluster_name, save_dir, members_df, prob_threshold, noise_df=None, candidates_df=None,
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
                                    label=f'candidates', s=dot_size, alpha=alpha, c=candidate_color, rasterized=True,
                                    cmap=cmap)
            if mean_predictions is not None:
                c_bar = fig.colorbar(colored_ax, ax=ax)
                c_bar.set_label("Membership probability", fontsize=16)
        if member_candidates_df is not None:
            ax.scatter(member_candidates_df[dic['x']], member_candidates_df[dic['y']], marker='+',
                       label=f'candidate = member (>{prob_threshold})', s=15 * dot_size, c='orange', rasterized=True)
        if non_member_candidates_df is not None:
            ax.scatter(non_member_candidates_df[dic['x']], non_member_candidates_df[dic['y']],
                       label=f'candidate = non member', s=dot_size, c='red', rasterized=True)
        ax.scatter(members_df[dic['x']], members_df[dic['y']],
                   label=f'members (>{prob_threshold})', s=5 * dot_size, c='blue', rasterized=True)

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


def plot_density_profile(cluster_name, save_dir, members, prob_threshold, cluster_kwargs=None, run_suffix=None,
                         plot=True, save=True):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if run_suffix is not None:
        suffix = '_' + run_suffix
    else:
        suffix = ''

    if type(members) == list:
        rs, densities, sigmas, _ = make_density_profile(members[0], cluster_kwargs)
        ax.errorbar(rs, densities, sigmas, fmt="o", capsize=3.0, label='old members')
        rs, densities, sigmas, areas = make_density_profile(members[1], cluster_kwargs)
        r_fit, density_fit, r_t = fit_king_model(rs, densities, areas)
        for i in range(len(r_fit) - 1):
            steepness = np.abs((np.log(density_fit[i+1]) - np.log(density_fit[i])) / (r_fit[i+1] - r_fit[i]))
            if steepness < 0.01:
                r_t = r_fit[i]
                break
        ax.plot(r_fit, density_fit, label=r'king model fit ($R_{t}$' + f'={np.round(r_t, 1)} pc)')
        ax.errorbar(rs, densities, sigmas, fmt="o", capsize=3.0, label='old + new members')
        ax.legend()
    else:
        rs, densities, sigmas, areas = make_density_profile(members, cluster_kwargs)
        x, f_x, r_t = fit_king_model(rs, densities, areas)
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
                    x_fields, y_fields, dot_size=4.0, run_suffix=None, plot=True, save=True):
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

        # if y == 'parallax':
        #     ax.errorbar(new_members[x], new_members[y], yerr=new_members['parallax_error'],
        #                 fmt='.', label=f'new members (>{member_prob_threshold})', markersize=0.4 * dot_size, c='red')
        #     ax.errorbar(old_members[x], old_members[y], yerr=old_members['parallax_error'],
        #                 fmt='.', label=f'old members (>{member_prob_threshold})', markersize=0.2 * dot_size, c='blue')
        #     ax.errorbar(members_to_compare_to[x], members_to_compare_to[y], yerr=members_to_compare_to['parallax_error'],
        #                 fmt='.', label=f'compare members (>{member_prob_threshold})', markersize=0.1 * dot_size, c='green')
        # else:
        ax.scatter(new_members[x], new_members[y], label=f'new members (>{member_prob_threshold})', s=4 * dot_size,
                   c='red')
        ax.scatter(old_members[x], old_members[y], label=f'old members (>{member_prob_threshold})', s=2 * dot_size,
                   c='blue')
        ax.scatter(members_to_compare_to[x], members_to_compare_to[y],
                   label=f'compare members (>{member_prob_threshold})', s=dot_size, c='green')
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


def make_plots(cluster_name, save_dir, cluster_kwargs, plot_prob_threshold, members, candidates, member_candidates,
               members_compare, noise, isochrone, mean_predictions, suffix, plot, save_plots):
    old_members = members[members['PMemb'] >= plot_prob_threshold].copy()
    new_members = pd.concat((old_members, member_candidates), sort=False, ignore_index=True)

    r_t = plot_density_profile(cluster_name, save_dir, [members, new_members], plot_prob_threshold, cluster_kwargs,
                               run_suffix=suffix, plot=plot, save=save_plots)

    member_candidates = crop_member_candidates(member_candidates, r_t, plot_prob_threshold, cluster_kwargs)
    new_members = pd.concat((old_members, member_candidates), sort=False, ignore_index=True)

    plot_mass_segregation_profile(cluster_name, save_dir, [members, new_members], plot_prob_threshold, cluster_kwargs,
                                  run_suffix=suffix, plot=plot, save=save_plots)

    plot_sources(cluster_name, save_dir, members, plot_prob_threshold, candidates_df=candidates, noise_df=noise,
                 isochrone_df=isochrone, mean_predictions=mean_predictions, zoom=1.5, dot_size=0.5, alpha=1.0,
                 cmap='autumn_r', run_suffix=suffix, plot=plot, save=save_plots)
    plot_sources(cluster_name, save_dir, members, plot_prob_threshold, member_candidates_df=member_candidates,
                 mean_predictions=mean_predictions, isochrone_df=isochrone, zoom=1.0, dot_size=1.0, alpha=1.0,
                 cmap='autumn_r', run_suffix=suffix, plot=plot, save=save_plots)

    if not members_compare.empty:
        compare_members(cluster_name, save_dir, old_members, member_candidates, members_compare,
                        plot_prob_threshold, x_fields=['ra', 'bp_rp', 'pmra', 'phot_g_mean_mag'],
                        y_fields=['dec', 'phot_g_mean_mag', 'pmdec', 'parallax'], run_suffix=suffix, plot=plot,
                        save=save_plots)
