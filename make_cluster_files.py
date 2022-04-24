import os
import pandas as pd
import numpy as np

data_path = '/home/matthijs/git/gaia_oc_amd/data'

dataset = 'cg18'
# dataset = 't22'

if dataset == 'cg18':
    clusters_file = 'cg18_clusters.tsv'
    members_file = 'cg18_members.tsv'
    clusters_df = pd.read_csv(os.path.join(data_path, clusters_file), sep='\t', header=45)
    members_df = pd.read_csv(os.path.join(data_path, members_file), sep='\t', header=48, dtype=str)

    # the 'clusters.tsv' file contains the names of the clusters
    clusters_df = clusters_df.iloc[2:]
    clusters_df = clusters_df.reset_index(drop=True)

    # the 'members.tsv' file contains the members of all clusters in 'clusters.tsv' combined
    members_df = members_df.iloc[2:]
    members_df = members_df.reset_index(drop=True)

    members_df['Source'] = members_df['Source'].astype(np.int64)
    members_df['RA_ICRS'] = members_df['RA_ICRS'].astype(np.float64)
    members_df['DE_ICRS'] = members_df['DE_ICRS'].astype(np.float64)
    members_df['pmRA'] = members_df['pmRA'].astype(np.float64)
    members_df['pmDE'] = members_df['pmDE'].astype(np.float64)
    members_df['plx'] = members_df['plx'].astype(np.float64)
    members_df['Gmag'] = members_df['Gmag'].astype(np.float64)
    members_df['PMemb'] = members_df['PMemb'].astype(np.float32)

    members_df = members_df.rename(columns={'RA_ICRS': 'ra', 'DE_ICRS': 'dec', 'pmRA': 'pmra', 'pmDE': 'pmdec',
                                            'plx': 'parallax', 'Gmag': 'phot_g_mean_mag', 'BP-RP': 'bp_rp'})
    cluster_name_column = 'Cluster'
elif dataset == 't22':
    clusters_file = 't22_clusters.csv'
    members_file = 't22_members.csv'
    clusters_df = pd.read_csv(os.path.join(data_path, clusters_file))
    members_df = pd.read_csv(os.path.join(data_path, members_file), dtype=str)

    # the 'clusters.tsv' file contains the names of the clusters
    clusters_df = clusters_df.iloc[2:]
    clusters_df = clusters_df.reset_index(drop=True)

    # the 'members.tsv' file contains the members of all clusters in 'clusters.tsv' combined
    members_df = members_df.iloc[2:]
    members_df = members_df.reset_index(drop=True)

    members_df['source_id'] = members_df['source_id'].astype(np.int64)
    members_df['ra'] = members_df['ra'].astype(np.float64)
    members_df['dec'] = members_df['dec'].astype(np.float64)
    members_df['parallax'] = members_df['parallax'].astype(np.float64)
    members_df['phot_g_mean_mag'] = members_df['phot_g_mean_mag'].astype(np.float64)
    members_df['proba'] = members_df['proba'].astype(np.float32)

    members_df = members_df.rename(columns={'cluster': 'Cluster', 'source_id': 'Source', 'proba': 'PMemb'})
    cluster_name_column = 'oc'
else:
    raise ValueError(f'Unknown dataset {dataset}')

save_dir = os.path.join(data_path, 'compare_members')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

cluster_names = clusters_df[cluster_name_column].str.strip().values

for name in cluster_names:
    cluster_df = members_df[members_df['Cluster'] == name.strip()]
    cluster_df = cluster_df.reset_index(drop=True)[['Source', 'ra', 'dec', 'pmra', 'pmdec', 'parallax',
                                                    'phot_g_mean_mag', 'PMemb']]
    cluster_df.to_csv(os.path.join(save_dir, name.strip() + '.csv'))
    print(name, len(cluster_df))
