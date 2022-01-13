import matplotlib.pyplot as plt
import pandas as pd


def save_csv(members_df, noise_df, member_candidates_df, non_member_candidates_df, combined=True):
    if combined:
        members_df['id'] = 0
        noise_df['id'] = 1
        member_candidates_df['id'] = 2
        non_member_candidates_df['id'] = 3

        cone_df = pd.concat((members_df, noise_df, member_candidates_df, non_member_candidates_df),
                            sort=False, ignore_index=True)
        cone_df.to_csv('csv/cone.csv')
    else:
        members_df.to_csv('csv/members.csv')
        noise_df.to_csv('csv/noise.csv')
        member_candidates_df.to_csv('csv/member_candidates.csv')
        non_member_candidates_df.to_csv('csv/non_member_candidates.csv')


def make_plots(members_df, noise_df=None, candidates_df=None, member_candidates_df=None, non_member_candidates_df=None,
               isochrone_df=None, title='Cluster', zoom=1.0, dot_size=1.0, alpha=0.8, save=True):

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
        if non_member_candidates_df is not None:
            field_min = non_member_candidates_df[field].min()
            field_max = non_member_candidates_df[field].max()
        elif candidates_df is not None:
            field_min = candidates_df[field].min()
            field_max = candidates_df[field].max()
        else:
            field_min = members_df[field].min()
            field_max = members_df[field].max()
        field_range = field_max - field_min
        center = field_min + field_range / 2
        plot_ranges.update({field: [center - field_range / zoom, center + field_range / zoom]})

    for dic in plot_kwargs:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        if noise_df is not None:
            ax.scatter(noise_df[dic['x']], noise_df[dic['y']],
                       label='non members', s=0.2 * dot_size, c='gray', alpha=0.2, rasterized=True)
        if candidates_df is not None:
            ax.scatter(candidates_df[dic['x']], candidates_df[dic['y']],
                       label=f'candidates', s=dot_size, c='orange', rasterized=True)
        if member_candidates_df is not None:
            ax.scatter(member_candidates_df[dic['x']], member_candidates_df[dic['y']],
                       label=f'candidate = member', s=dot_size, c='green', rasterized=True)
        if non_member_candidates_df is not None:
            ax.scatter(non_member_candidates_df[dic['x']], non_member_candidates_df[dic['y']],
                       label=f'candidate = non member', s=dot_size, c='red', rasterized=True)
        ax.scatter(members_df[dic['x']], members_df[dic['y']],
                   label=f'members', s=dot_size, c='blue', alpha=alpha, rasterized=True)

        if dic['x'] in plot_ranges:
            ax.set_xlim(plot_ranges[dic['x']][0], plot_ranges[dic['x']][1])
        if dic['y'] in plot_ranges:
            ax.set_ylim(plot_ranges[dic['y']][0], plot_ranges[dic['y']][1])
        if dic['y'] == 'phot_g_mean_mag':
            ax.plot(isochrone_df['bp_rp'], isochrone_df['phot_g_mean_mag'], label='isochrone')
            ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(dic['x'])
        ax.set_ylabel(dic['y'])
        ax.legend(fontsize='small')

        if save:
            plt.savefig(f"results/{dic['x']}_{dic['y']}.png")
        plt.show()


def print_sets(member_candidates_indices, lp_members, noise, candidates):
    # create various sets of source identities belonging to different groups
    candidate_ids = set(candidates['source_id'].to_numpy())
    noise_ids = set(noise['source_id'].to_numpy())
    lp_member_ids = set(lp_members['source_id'].to_numpy())
    member_candidates_ids = set(candidates['source_id'].to_numpy()[member_candidates_indices])

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
