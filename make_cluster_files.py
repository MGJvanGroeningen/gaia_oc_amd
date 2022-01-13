import os
import pandas as pd

data_path = '/home/matthijs/Documents/Studie/Master_Astronomy/2nd_Research_Project/data'

# the 'clusters.tsv' file contains the names of the clusters (~1400)
clusters_file = os.path.join(data_path, 'clusters.tsv')
clusters_df = pd.read_csv(clusters_file, sep='\t', header=45)
clusters_df = clusters_df.iloc[2:]
clusters_df = clusters_df.reset_index(drop=True)

# the 'cluster_members.tsv' file contains the members of all clusters in 'clusters.tsv' combined
members_file = os.path.join(data_path, 'cluster_members.tsv')
members_df = pd.read_csv(members_file, sep='\t', header=36, dtype=str)
members_df = members_df.iloc[2:]
members_df = members_df.reset_index(drop=True)

save_dir = os.path.join(data_path, 'clusters')
cluster_names = clusters_df['Cluster'].values

for name in cluster_names:
    cluster_df = members_df[members_df['Cluster'] == name].reset_index(drop=True).drop(['Cluster', 'SimbadName'], axis=1)
    cluster_df.to_csv(os.path.join(save_dir, name.strip() + '.csv'))
