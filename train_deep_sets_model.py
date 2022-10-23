import os
import argparse
import random
from torch.utils.data import DataLoader

from gaia_oc_amd.io import save_hyper_parameters, cluster_list, save_metrics
from gaia_oc_amd.utils import property_mean_and_std

from gaia_oc_amd.data_preparation.datasets import multi_cluster_deep_sets_dataset, DeepSetsDataset

from gaia_oc_amd.machine_learning.training import train_model
from gaia_oc_amd.machine_learning.deepsets_zaheer import D5

from gaia_oc_amd.candidate_evaluation.visualization import plot_metrics


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
    parser.add_argument('--training_features', nargs='*', type=str, default=['f_r', 'f_pm', 'f_plx', 'f_c', 'f_g'],
                        help='Features on which the model will be trained.')
    parser.add_argument('--hidden_size', nargs='?', type=int, default=64,
                        help='Hidden size of the neural network layers.')
    parser.add_argument('--load_model', nargs='?', type=bool, default=False,
                        help='Whether to load the parameters of an already trained model, saved at "save_path".')
    parser.add_argument('--n_epochs', nargs='?', type=int, default=100,
                        help='The number of training epochs.')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-6,
                        help='Learning rate.')
    parser.add_argument('--l2', nargs='?', type=float, default=1e-5,
                        help='L2 regularization.')
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

    # main arguments
    clusters_dir = args_dict['clusters_dir']
    cluster_names = cluster_list(args_dict['cluster_names'])

    # path arguments
    model_dir = args_dict['model_dir']

    # data arguments
    validation_fraction = args_dict['validation_fraction']
    size_support_set = args_dict['size_support_set']
    batch_size = args_dict['batch_size']
    n_pos_duplicates = args_dict['n_pos_duplicates']
    neg_pos_ratio = args_dict['neg_pos_ratio']
    training_features = args_dict['training_features']

    # model arguments
    hidden_size = args_dict['hidden_size']
    load_model = args_dict['load_model']

    # training arguments
    n_epochs = args_dict['n_epochs']
    lr = args_dict['lr']
    l2 = args_dict['l2']
    weight_imbalance = args_dict['weight_imbalance']
    early_stopping_threshold = args_dict['early_stopping_threshold']
    seed = args_dict['seed']

    # plot arguments
    show = args_dict['show']

    # save arguments
    save_plot = args_dict['save_plot']

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # Create the model
    model = D5(hidden_size, x_dim=2 * len(training_features), pool='mean', out_dim=2)

    # Store feature means and standard deviations for future normalization during candidate evaluation
    training_feature_means, training_feature_stds = property_mean_and_std(clusters_dir, cluster_names,
                                                                          training_features)

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

    hyperparameters = {'n_training_clusters': len(train_clusters),
                       'training_clusters': sorted(train_clusters),
                       'n_validation_clusters': len(val_clusters),
                       'validation_clusters': sorted(val_clusters),
                       'validation_fraction': validation_fraction,
                       'size_support_set': size_support_set,
                       'batch_size': batch_size,
                       'n_pos_duplicates': n_pos_duplicates,
                       'neg_pos_ratio': neg_pos_ratio,
                       'training_features': training_features,
                       'training_feature_means': list(training_feature_means),
                       'training_feature_stds': list(training_feature_stds),
                       'hidden_size': hidden_size,
                       'n_epochs': n_epochs,
                       'lr': lr,
                       'l2': l2,
                       'weight_imbalance': weight_imbalance,
                       'early_stopping_threshold': early_stopping_threshold,
                       'seed': seed}

    save_hyper_parameters(model_dir, hyperparameters)

    print('Number of validation clusters:', n_val_clusters)
    print('Validation clusters:', sorted(val_clusters))
    print('Training clusters:', sorted(train_clusters))

    # Create training and validation datasets
    train_dataset = multi_cluster_deep_sets_dataset(clusters_dir, train_clusters, training_features,
                                                    training_feature_means, training_feature_stds,
                                                    n_pos_duplicates=n_pos_duplicates, neg_pos_ratio=neg_pos_ratio,
                                                    size_support_set=size_support_set, seed=seed)
    train_dataset = DataLoader(DeepSetsDataset(train_dataset), batch_size=batch_size, shuffle=True)
    if len(val_clusters) > 0:
        val_dataset = multi_cluster_deep_sets_dataset(clusters_dir, val_clusters, training_features,
                                                      training_feature_means, training_feature_stds,
                                                      n_pos_duplicates=n_pos_duplicates, neg_pos_ratio=neg_pos_ratio,
                                                      size_support_set=size_support_set, seed=seed)
        val_dataset = DataLoader(DeepSetsDataset(val_dataset), batch_size=batch_size, shuffle=True)
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
