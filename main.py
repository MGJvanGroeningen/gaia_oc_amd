import os
from torch import load

from gaia_oc_amd.data_preparation.isochrone import make_isochrone
from gaia_oc_amd.data_preparation.features import Features
from gaia_oc_amd.data_preparation.parse_cone import parse_cone
from gaia_oc_amd.data_preparation.cones_query import download_cones
from gaia_oc_amd.data_preparation.datasets import deep_sets_datasets
from gaia_oc_amd.candidate_evaluation.probabilities import candidate_probabilities
from gaia_oc_amd.data_preparation.sets import Sources
from gaia_oc_amd.data_preparation.cluster import Cluster
from gaia_oc_amd.neural_networks.training import train_model
from gaia_oc_amd.neural_networks.deepsets_zaheer import D5
from gaia_oc_amd.candidate_evaluation.visualization import plot_sources, make_plots, plot_loss_accuracy
from gaia_oc_amd.data_preparation.utils import save_sets, load_sets, save_tidal_radius, load_cluster_parameters, \
    load_cone, load_members


if __name__ == "__main__":
    cluster_names = ['NGC_752']

    print(cluster_names)
    print('Number of clusters:', len(cluster_names))

    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    cluster_path = os.path.join(data_dir, 'cluster_parameters.tsv')
    clusters = [Cluster(load_cluster_parameters(cluster_name, cluster_path)) for cluster_name in cluster_names]

    # credentials_path = os.path.join(data_dir, 'gaia_credentials')
    # download_cones(clusters, data_dir, credentials_path)

    isochrone_dir = os.path.join(data_dir, 'isochrones')
    prob_threshold = 1.0

    # plot params
    show = True
    save = True
    show_features = False

    for cluster in clusters:
        cone_path = os.path.join(data_dir, 'clusters', cluster.name, 'cone.vot.gz')
        members_path = os.path.join(data_dir, 'cg18_members.tsv')
        comparison_path = os.path.join(data_dir, 't22_members.tsv')

        cone = load_cone(cone_path, cluster)
        isochrone = make_isochrone(isochrone_dir, cluster)

        members = load_members(members_path, cluster.name, cluster_column='Cluster', id_column='Source')

        hp_members = members[members['PMemb'] >= prob_threshold]
        while len(hp_members) < 15:
            prob_threshold -= 0.1
            hp_members = members[members['PMemb'] >= prob_threshold]

        member_ids = hp_members['source_id']
        member_probs = hp_members['PMemb']

        comparison = load_members(comparison_path, cluster.name, cluster_column='Cluster', id_column='GaiaEDR3',
                                  prob_column='Proba')

        print(comparison)
        exit(0)

        if len(comparison) > 0:
            comparison_ids = comparison['source_id']
            comparison_probs = comparison['PMemb']
        else:
            comparison_ids = None
            comparison_probs = None

        members, candidates, non_members, comparison = parse_cone(cluster, cone, isochrone, member_ids,
                                                                  member_probs=member_probs,
                                                                  comparison_ids=comparison_ids,
                                                                  comparison_probs=comparison_probs)

        sources = Sources(members, candidates, non_members, comparison_members=comparison)
        save_dir = os.path.join(data_dir, 'clusters', cluster.name)
        plot_sources(sources, cluster, save_dir, isochrone=isochrone, plot_type='candidates', show=show, save=save)

        print('Members:', len(members))
        print('Candidates:', len(candidates))
        print('Non-members:', len(non_members))

        save_sets(data_dir, cluster, members, candidates, non_members, comparison)

    cluster_names = ['NGC_2509']
    data_dir = os.path.join(os.getcwd(), 'data')

    # data params
    validation_fraction = 0.3
    size_support_set = 5
    batch_size = 32
    max_members = 2000
    max_non_members = 20000
    training_features = ['f_r', 'f_pm', 'f_plx', 'f_c', 'f_g']

    # model params
    hidden_size = 64

    # training params
    n_epochs = 20
    lr = 1e-5
    l2 = 1e-5
    weight_imbalance = 2.
    load_cp = False

    # plot params
    show = True
    save_plots = True

    model = D5(hidden_size, x_dim=2 * len(training_features), pool='mean', out_dim=2)

    train_dataset, val_dataset = deep_sets_datasets(data_dir, cluster_names, training_features, validation_fraction,
                                                    max_members=max_members, max_non_members=max_non_members,
                                                    size_support_set=size_support_set, seed=42)

    metrics = train_model(model, train_dataset, val_dataset, save_path='deep_sets_model', num_epochs=n_epochs,
                          lr=lr, l2=l2, weight_imbalance=weight_imbalance, load_checkpoint=load_cp)
    plot_loss_accuracy(metrics, os.getcwd(), show, save_plots)

    # data params
    cluster_names = ['NGC_2509']
    size_support_set = 5
    training_features = ['f_r', 'f_pm', 'f_plx', 'f_c', 'f_g']

    # model params
    hidden_size = 64

    data_dir = os.path.join(os.getcwd(), 'data')
    isochrone_dir = os.path.join(data_dir, 'isochrones')

    model = D5(hidden_size, x_dim=2 * len(training_features), pool='mean', out_dim=2)
    model.load_state_dict(load('deep_sets_model'))

    # evaluation params
    n_samples = 40

    # plot params
    show = True
    save = True
    show_features = False
    prob_threshold = 1.0

    for cluster_name in cluster_names:
        cluster, members, candidates, non_members, comparison = load_sets(data_dir, cluster_name)
        isochrone = make_isochrone(isochrone_dir, cluster)

        save_dir = os.path.join(data_dir, 'clusters', cluster.name)

        features = Features(training_features, cluster, isochrone)
        sources = Sources(members, candidates, non_members, comparison_members=comparison)

        plot_sources(sources, cluster, save_dir, isochrone=isochrone, plot_type='candidates',
                     prob_threshold=prob_threshold, show=show, save=save, show_features=show_features)

        sources.candidates['PMemb'] = candidate_probabilities(model, sources, features, n_samples, size_support_set)
        sources.candidates.to_csv(os.path.join(save_dir, 'candidates.csv'))

        save_tidal_radius(data_dir, sources.candidates.hp(min_prob=0.1), cluster)

        if show or save:
            make_plots(sources, cluster, save_dir, isochrone, prob_threshold=prob_threshold, show=show,
                       save=save)
