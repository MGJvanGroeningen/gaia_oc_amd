import os
import pandas as pd


def save_csv(cluster_save_dir, hp_members_df, lp_members_df, noise_df, member_candidates_df, non_member_candidates_df,
             compare_df, run_suffix=None, combined=True):

    if run_suffix is not None:
        suffix = '_' + run_suffix
    else:
        suffix = ''

    sources = [hp_members_df, lp_members_df, member_candidates_df, non_member_candidates_df, noise_df]
    ids = [0, 1, 2, 3, 4]
    source_labels = ['hp_members', 'lp_members', 'member_candidates', 'non_member_candidates', 'noise']

    cone_save_file = os.path.join(cluster_save_dir, 'cone' + suffix + '.csv')

    if combined:
        for source, idx in zip(sources, ids):
            source['id'] = idx

        cone_df = pd.concat(sources, sort=False, ignore_index=True)
        cone_df.to_csv(cone_save_file)
    else:
        for source, label in zip(sources, source_labels):
            source.to_csv(os.path.join(cluster_save_dir, f'{label}{suffix}.csv'))

    compare_save_file = os.path.join(cluster_save_dir, 'compare' + suffix + '.csv')
    compare_df.to_csv(compare_save_file)


def load_csv(cluster_name, save_dir, run_suffix=None):
    if run_suffix is not None:
        suffix = '_' + run_suffix
    else:
        suffix = ''

    cone_file = os.path.join(save_dir, 'results', cluster_name, f'cone{suffix}.csv')
    compare_file = os.path.join(save_dir, 'results', cluster_name, f'compare{suffix}.csv')

    cone_df = pd.read_csv(cone_file)
    compare_members = pd.read_csv(compare_file)

    hp_members = cone_df[cone_df['id'] == 0].copy()
    lp_members = cone_df[cone_df['id'] == 1].copy()
    member_candidates = cone_df[cone_df['id'] == 2].copy()
    non_member_candidates = cone_df[cone_df['id'] == 3].copy()
    noise = cone_df[cone_df['id'] == 4].copy()

    candidates = pd.concat((member_candidates, non_member_candidates), axis=0)

    parsed_sources = {'hp_members': hp_members,
                      'lp_members': lp_members,
                      'compare_members': compare_members,
                      'noise': noise,
                      'candidates': candidates,
                      'member_candidates': member_candidates,
                      'non_member_candidates': non_member_candidates}
    return parsed_sources
