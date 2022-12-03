import os
import argparse
import random
from torch.utils.data import DataLoader

from gaia_oc_amd.io import cluster_list, save_metrics

from gaia_oc_amd.data_preparation.datasets import property_mean_and_std, MultiClusterDeepSetsDataset

from gaia_oc_amd.machine_learning.training import train_model
from gaia_oc_amd.machine_learning.deepsets_zaheer import D5

from gaia_oc_amd.candidate_evaluation.visualization import plot_metrics


def train_deep_sets_model(cluster_names, clusters_dir='./clusters', model_dir='./deep_sets_model',
                          validation_fraction=0.3, size_support_set=10, batch_size=32, n_pos_duplicates=2,
                          neg_pos_ratio=5, source_features=('f_r', 'f_pm', 'f_plx', 'f_c', 'f_g'),
                          cluster_features=('a0', 'age', 'parallax'), hidden_size=64, load_model=False, n_epochs=100,
                          lr=1e-6, l2=1e-5, weight_imbalance=5., early_stopping_threshold=20, seed=42, show=False,
                          save_plot=True):
    """Main function for training a deep sets model on the members and non-members of a list of clusters. This function
    contains the following steps:
        - Create a normalized training and validation set from the members and non-members of the supplied clusters.
        - Create a 'hyper_parameter' file wich contains the relevant hyper parameters of the model and the training
            process.
        - Train the model
        - Save and optionally plot the metrics which were kept track of during training

    Args:
        cluster_names (str, list): 'Names of the open cluster(s) we want to build sets for. Can be a name or a file
            with cluster names.'
        clusters_dir (str): 'Directory where cluster data (e.g. cone searches, source sets) and results will be saved.'
        model_dir (str): 'Directory where the model parameters will be saved.'
        validation_fraction (float): 'Fraction of the data to use for validation.'
        size_support_set (int): 'The number of members in the support set.'
        batch_size (int): 'Batch size of the training and validation datasets.'
        n_pos_duplicates (int):'Number of times a positive example is included in the dataset (with different support
            set).'
        neg_pos_ratio (int): 'Number of negative examples included in the dataset per positive example.'
        source_features (str, list): 'Source eatures on which the model will be trained.'
        cluster_features (str, list): 'Cluster features on which the model will be trained.'
        hidden_size (int): 'Size of the hidden neural network layers.'
        load_model (bool): 'Whether to load the parameters of an already trained model, saved at "save_path".'
        n_epochs (int): 'The number of training epochs.'
        lr (float): 'Learning rate.'
        l2 (float): 'L2 regularization (weight decay).'
        weight_imbalance (float): 'Weight imbalance. A higher value gives more weight to classifying members as opposed
            to non-members.'
        early_stopping_threshold (int): 'Number of epochs after which the training is terminated if the model has not
            improved.'
        seed (int): 'The seed that determines the distribution of train and validation data.'
        show (bool): 'Whether to show the candidates plot.'
        save_plot (bool): 'Whether to save the candidates plot.'

    """
    cluster_names = cluster_list(cluster_names)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    source_features = list(source_features)
    cluster_features = list(cluster_features)

    # Create the model
    model = D5(hidden_size, x_dim=2 * len(source_features) + len(cluster_features), pool='mean', out_dim=2)

    # Store feature means and standard deviations for future normalization during candidate evaluation
    means, stds = property_mean_and_std(clusters_dir, cluster_names, source_features, cluster_features)

    n_clusters = len(cluster_names)
    if n_clusters == 1:
        train_clusters = cluster_names
        val_clusters = []
        n_val_clusters = 0
    else:
        random.seed(seed)
        n_val_clusters = min(max(int(n_clusters * validation_fraction), 1), n_clusters - 1)
        val_clusters = random.sample(cluster_names, n_val_clusters)
        train_clusters = list(set(cluster_names) - set(val_clusters))

    print('Number of validation clusters:', n_val_clusters)
    print('Validation clusters:', sorted(val_clusters))
    print('Training clusters:', sorted(train_clusters))

    # Create training and validation datasets
    train_dataset = MultiClusterDeepSetsDataset(clusters_dir, train_clusters, source_features,
                                                source_feature_means=means['source'],
                                                source_feature_stds=stds['source'],
                                                cluster_feature_names=cluster_features,
                                                cluster_feature_means=means['cluster'],
                                                cluster_feature_stds=stds['cluster'],
                                                n_pos_duplicates=n_pos_duplicates,
                                                neg_pos_ratio=neg_pos_ratio, size_support_set=size_support_set,
                                                seed=seed)
    train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if len(val_clusters) > 0:
        val_dataset = MultiClusterDeepSetsDataset(clusters_dir, val_clusters, source_features,
                                                  source_feature_means=means['source'],
                                                  source_feature_stds=stds['source'],
                                                  cluster_feature_names=cluster_features,
                                                  cluster_feature_means=means['cluster'],
                                                  cluster_feature_stds=stds['cluster'],
                                                  n_pos_duplicates=n_pos_duplicates,
                                                  neg_pos_ratio=neg_pos_ratio, size_support_set=size_support_set,
                                                  seed=seed)
        val_dataset = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    else:
        val_dataset = None

    # Train the model
    print(' ')
    metrics = train_model(model, model_dir, train_dataset, val_dataset=val_dataset, num_epochs=n_epochs, lr=lr,
                          l2=l2, weight_imbalance=weight_imbalance, early_stopping_threshold=early_stopping_threshold,
                          load_model=load_model)
    print(f'Saved model parameters in {os.path.abspath(model_dir)}')

    save_metrics(model_dir, metrics)

    # Show training progress
    plot_metrics(metrics, model_dir, show=show, save=save_plot)
    if save_plot:
        print(f'Created loss and accuracy plot at {os.path.abspath(os.path.join(model_dir, "metrics.png"))}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('cluster_names', nargs='?', type=str,
                        help='Names of the open cluster(s) we want to build sets for. '
                             'Can be a name or a file with cluster names.')
    parser.add_argument('--clusters_dir', nargs='?', type=str, default='clusters',
                        help='Directory where cluster data (i.e. cluster properties, source sets) is saved.')
    parser.add_argument('--model_dir', nargs='?', type=str, default='deep_sets_model',
                        help='Directory where the model parameters will be saved.')
    parser.add_argument('--validation_fraction', nargs='?', type=float, default=0.3,
                        help='Fraction of the data to use for validation.')
    parser.add_argument('--size_support_set', nargs='?', type=int, default=10,
                        help='The number of members in the support set.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=32,
                        help='Batch size of the training and validation datasets.')
    parser.add_argument('--n_pos_duplicates', nargs='?', type=int, default=2,
                        help='Number of times a positive example is included '
                             'in the dataset (with different support set).')
    parser.add_argument('--neg_pos_ratio', nargs='?', type=int, default=5,
                        help='Number of negative examples included in the dataset per positive example.')
    parser.add_argument('--source_features', nargs='*', type=str, default=['f_r', 'f_pm', 'f_plx', 'f_c', 'f_g'],
                        help='Source features on which the model will be trained.')
    parser.add_argument('--cluster_features', nargs='*', type=str, default=['a0', 'age', 'parallax'],
                        help='Cluster features on which the model will be trained.')
    parser.add_argument('--hidden_size', nargs='?', type=int, default=64,
                        help='Hidden size of the neural network layers.')
    parser.add_argument('--load_model', nargs='?', type=bool, default=False,
                        help='Whether to load the parameters of an already trained model, saved at "save_path".')
    parser.add_argument('--n_epochs', nargs='?', type=int, default=100,
                        help='The number of training epochs.')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-5,
                        help='Learning rate.')
    parser.add_argument('--l2', nargs='?', type=float, default=1e-5,
                        help='L2 regularization (weight decay).')
    parser.add_argument('--weight_imbalance', nargs='?', type=float, default=5.,
                        help='Weight imbalance. '
                             'A higher value gives more weight to classifying members as opposed to non-members.')
    parser.add_argument('--early_stopping_threshold', nargs='?', type=int, default=20,
                        help='Number of epochs after which the training is terminated if the model has not improved.')
    parser.add_argument('--seed', nargs='?', type=int, default=42,
                        help='The seed that determines the distribution of train and validation data.')
    parser.add_argument('--show', nargs='?', type=bool, default=False,
                        help='Whether to show the loss and accuracy plot.')
    parser.add_argument('--save_plot', nargs='?', type=bool, default=True,
                        help='Whether to save the loss and accuracy plot.')

    args_dict = vars(parser.parse_args())

    train_deep_sets_model(args_dict['cluster_names'],
                          clusters_dir=args_dict['clusters_dir'],
                          model_dir=args_dict['model_dir'],
                          validation_fraction=args_dict['validation_fraction'],
                          size_support_set=args_dict['size_support_set'],
                          batch_size=args_dict['batch_size'],
                          n_pos_duplicates=args_dict['n_pos_duplicates'],
                          neg_pos_ratio=args_dict['neg_pos_ratio'],
                          source_features=args_dict['source_features'],
                          cluster_features=args_dict['cluster_features'],
                          hidden_size=args_dict['hidden_size'],
                          load_model=args_dict['load_model'],
                          n_epochs=args_dict['n_epochs'],
                          lr=args_dict['lr'],
                          l2=args_dict['l2'],
                          weight_imbalance=args_dict['weight_imbalance'],
                          early_stopping_threshold=args_dict['early_stopping_threshold'],
                          seed=args_dict['seed'],
                          show=args_dict['show'],
                          save_plot=args_dict['save_plot'])
