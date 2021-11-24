import matplotlib.pyplot as plt
from filter_cone_data import make_non_member_df, make_high_prob_member_df, plx_distance_rule, pm_distance_rule, \
    isochrone_distance_rule, make_candidate_rule
from rf_cone_data_advanced_practice import load_cone_data, load_cluster_data


def plot_member_non_member(members,
                           non_members,
                           x_name,
                           y_name,
                           probability_threshold,
                           plot_name='cluster',
                           dot_size=1.0,
                           alpha=1.0):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.scatter(non_members[x_name], non_members[y_name], label='non members', s=0.4 * dot_size, c='gray', alpha=alpha,
               rasterized=True)
    ax.scatter(members[x_name], members[y_name], label=f'members (>= {probability_threshold})', s=dot_size, c='blue', rasterized=True)
    ax.set_title(plot_name)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.legend(fontsize='small')
    return ax


def show_train_data(data_dir, 
                    cone_file, 
                    cluster_file,
                    candidate_selection_columns,
                    probability_threshold,
                    plx_sigma=3.0,
                    gmag_max_d=0.2,
                    pm_max_d=0.5,
                    bp_rp_max_d=0.05,
                    combined=False):
    cone_df = load_cone_data(data_dir, cone_file)
    cluster_df = load_cluster_data(data_dir, cluster_file)

    member_df = make_high_prob_member_df(cone_df, cluster_df, candidate_selection_columns, probability_threshold)

    gmag_plx_rule = plx_distance_rule(member_df, gmag_max_d, plx_sigma)
    pm_pm_rule = pm_distance_rule(member_df, pm_max_d, pm_max_d)
    bp_rp_gmag_rule = isochrone_distance_rule(member_df, bp_rp_max_d, gmag_max_d)
    combined_rule = make_candidate_rule(member_df, plx_sigma, gmag_max_d, pm_max_d, bp_rp_max_d)
    no_rule = lambda x: False

    if combined:
        plot_kwargs = [{'rule': gmag_plx_rule,
                        'x': 'ra',
                        'y': 'DEC'},
                       {'rule': pm_pm_rule,
                        'x': 'phot_g_mean_mag',
                        'y': 'parallax'},
                       {'rule': bp_rp_gmag_rule,
                        'x': 'pmra',
                        'y': 'pmdec'},
                       {'rule': no_rule,
                        'x': 'bp_rp',
                        'y': 'phot_g_mean_mag'}]
    else:
        plot_kwargs = [{'rule': gmag_plx_rule,
                        'x': 'phot_g_mean_mag',
                        'y': 'parallax'},
                       {'rule': pm_pm_rule,
                        'x': 'pmra',
                        'y': 'pmdec'},
                       {'rule': bp_rp_gmag_rule,
                        'x': 'bp_rp',
                        'y': 'phot_g_mean_mag'}]

    plot_ranges = {'phot_g_mean_mag': [10.5, 18.5],
                   'parallax': [-0.5, 1.25],
                   'pmra': [-3.7, -1.7],
                   'pmdec': [-0.2, 1.8],
                   'bp_rp': [0.0, 2.0]}

    for dic in plot_kwargs:
        non_member_df = make_non_member_df(cone_df,
                                           member_df,
                                           candidate_selection_columns,
                                           dic['rule'])

        print('members:', member_df.shape[0])
        print('non members:', non_member_df.shape[0])

        # ax = plot_member_non_member(member_df,
        #                             non_member_df,
        #                             dic['x'],
        #                             dic['y'],
        #                             probability_threshold,
        #                             plot_name=f'NGC 2509 with members (>= {probability_threshold})',
        #                             dot_size=5.0)
        # if dic['x'] in plot_ranges:
        #     ax.set_xlim(plot_ranges[dic['x']][0], plot_ranges[dic['x']][1])
        # if dic['y'] in plot_ranges:
        #     ax.set_ylim(plot_ranges[dic['y']][0], plot_ranges[dic['y']][1])
        # if dic['y'] == 'phot_g_mean_mag':
        #     ax.invert_yaxis()
        # if dic['y'] == 'parallax':
        #     ax.errorbar(member_df['phot_g_mean_mag'], member_df['parallax'], yerr=member_df['parallax_error'],
        #                 fmt='.', c='blue', elinewidth=0.5, rasterized=True)
        # plt.savefig(f"results/{dic['x']}_{dic['y']}.png")
        # plt.show()


if __name__ == "__main__":
    candidate_select_columns = ['ra', 'DEC', 'parallax', 'parallax_error', 'pmra', 'pmdec', 'phot_g_mean_mag', 'bp_rp',
                                'ruwe']

    show_train_data(data_dir='practice_data',
                    cone_file='NGC_2509_cone.csv',
                    cluster_file='NGC_2509.tsv',
                    candidate_selection_columns=candidate_select_columns,
                    probability_threshold=0.5,
                    plx_sigma=3.0,
                    gmag_max_d=0.2,
                    pm_max_d=0.5,
                    bp_rp_max_d=0.05,
                    combined=True)
