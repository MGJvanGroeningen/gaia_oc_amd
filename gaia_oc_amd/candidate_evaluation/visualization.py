import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from matplotlib_venn import venn2

from gaia_oc_amd.candidate_evaluation.diagnostics import make_density_profile, make_mass_segregation_profile, \
    fit_king_model

from gaia_oc_amd.data_preparation.features import isochrone_features_function


def plot_metrics(metrics, save_dir, loss_ylim=None, metric_ylim=None, show=True, save=True):
    """Plots the evolution of a number of metrics during training.

    Args:
        metrics (dict): Dictionary containing a number of metrics that indicate performance (loss,
            precision, recall, selectivity, accuracy, balanced accuracy and f1-score)
        save_dir (str): Directory where the plot is saved
        loss_ylim (tuple, float): Loss plot y-axis limits
        metric_ylim (tuple, float): Metrics plot y-axis limits
        show (bool): Whether to show the plot
        save (bool): Whether to save the plot

    """
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    fig.tight_layout(pad=6.0)

    epochs = np.arange(len(metrics['train_loss']))

    for mode, fmt in zip(['train', 'val'], ['-', '--']):
        metric = metrics[f'{mode}_loss']
        if len(metric) > 0:
            ax[0].plot(epochs, metric, label=mode, linestyle=fmt, c='tab:blue')

    for metr, col in zip(['precision', 'recall', 'selectivity', 'accuracy', 'balanced_accuracy', 'f1'],
                         ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']):
        for mode, fmt in zip(['train', 'val'], ['-', '--']):
            metric = metrics[f'{mode}_{metr}']
            if len(metric) > 0:
                ax[1].plot(epochs, metric, label=mode + ' ' + metr.replace('_', ' '), linestyle=fmt, c=col)
    
    labels = ['loss', 'metric']
    titles = ['Loss', 'Metrics']
    legend_fontsize = [10, 10]
    
    for i in range(2):
        ax[i].set_xticks(epochs[::5])
        ax[i].set_xlabel('epoch', fontsize=16)
        ax[i].set_ylabel(labels[i], fontsize=16)
        ax[i].set_title(titles[i], fontsize=20)
        ax[i].tick_params(labelsize=12)
        ax[i].grid(True)
        ax[i].legend(fontsize=legend_fontsize[i])
    if loss_ylim is not None:
        ax[0].set_ylim(loss_ylim)
    if metric_ylim is not None:
        ax[1].set_ylim(metric_ylim)

    if save:
        plt.savefig(os.path.join(save_dir, 'metrics.png'))
    if show:
        plt.show()
    plt.clf()
    plt.close('all')


def plot_density_profile(members, cluster, comparison=None, save_file='density_profile.png', members_label='members',
                         comparison_label='comparison', title='Density profile', show=True, save=False):
    """Plots a surface stellar density profile for the cluster sources.

    Args:
        members (Dataframe, list): Dataframe with members
        cluster (Cluster): Cluster object
        comparison (Dataframe, list): Another dataframe with members
        save_file (str): Path where the plot is saved
        members_label (str): Label to identify the member set
        comparison_label (str): Label to identify the comparison set
        title (Cluster): Plot title
        show (bool): Whether to show the plot
        save (bool): Whether to save the plot

    """
    members_sets = [members, comparison]
    labels = [members_label, comparison_label]
    colours = ['black', 'gray']
    zorders = [2, 0]

    if comparison is not None:
        n_plots = 2
    else:
        n_plots = 1

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for i in range(n_plots):
        mem, lab, col, zorder = members_sets[i], labels[i], colours[i], zorders[i]
        rs, areas, densities, sigmas = make_density_profile(mem, cluster)
        r_fit, density_fit, r_t = fit_king_model(rs, densities, areas)
        ax.errorbar(rs, densities, sigmas, fmt="o", capsize=3.0, label=f'Stellar surface density ({lab})', c=col,
                    zorder=zorder - 1)
        ax.plot(r_fit, density_fit, label=r'King model fit, $R_{t}$' + f'={np.round(r_t, 1)} pc ' + f'({lab})', c=col,
                zorder=zorder)

    ax.set_yscale('log')
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('r [pc]', fontsize=18)
    ax.set_ylabel(r'$\rho$ [stars / pc$^{2}$]', fontsize=18)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=12)
    
    if save:
        plt.savefig(save_file)
    if show:
        plt.show()
    plt.clf()
    plt.close('all')


def plot_mass_segregation_profile(members, cluster, comparison=None, save_file='mass_segregation.png',
                                  members_label='members', comparison_label='comparison', title='Mass segregation',
                                  min_n_mst=5, max_n_mst=100, n_samples=20, show=True, save=False):
    """Plots a mass segregation profile for the cluster sources.

    Args:
        members (Dataframe, list): Dataframe with members
        cluster (Cluster): Cluster object
        comparison (Dataframe, list): Another dataframe with members
        save_file (str): Path where the plot is saved
        members_label (str): Label to identify the member set
        comparison_label (str): Label to identify the comparison set
        title (str): Plot title
        min_n_mst (int): Minimum number of members for which to calculate the mass segregation ratio
        max_n_mst (int): Maximum number of members for which to calculate the mass segregation ratio
        n_samples (int): The number of samples of the random set of N members.
        show (bool): Whether to show the plot
        save (bool): Whether to save the plot

    """

    members_sets = [members, comparison]
    labels = [members_label, comparison_label]
    colours = ['black', 'gray']
    zorders = [1, 0]

    if comparison is not None:
        n_plots = 2
    else:
        n_plots = 1

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for i in range(n_plots):
        mem, lab, col, zorder = members_sets[i], labels[i], colours[i], zorders[i]
        n_mst, lambda_msr, sigmas = make_mass_segregation_profile(mem, cluster, min_n_mst, max_n_mst, n_samples)
        ax.errorbar(n_mst, lambda_msr, sigmas, elinewidth=0.5, ms=2.0, capsize=2.0, fmt="o", c=col,
                    label=f'Mass segregation ratio ({lab})', zorder=zorder)

    ax.hlines(1, min_n_mst, max_n_mst, linestyles='dashed')
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(r'$N_{MST}$', fontsize=18)
    ax.set_ylabel(r'$\Lambda_{MSR}$', fontsize=18)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=12)
    
    if save:
        plt.savefig(save_file)
    if show:
        plt.show()
    plt.clf()
    plt.close('all')


def plot_confusion_matrix(members1, members2, save_file='membership_comparison.png', title='Confusion matrix',
                          label1='members_1', label2='members_2', show=True, save=False):
    """Plots a 'confusion matrix' that compares the probabilities of the candidates and comparison/train members.

    Args:
        members1 (Dataframe): Dataframe containing members
        members2 (Dataframe): Another dataframe containing members
        save_file (str): Path where the plot is saved
        title (Cluster): Plot title
        label1 (string): Label of the first member set
        label2 (string): Label of the second member set
        show (bool): Whether to show the plot
        save (bool): Whether to save the plot

    """
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    merged = pd.merge(members2, members1, how='outer', on='source_id').fillna(0)
    probabilities = np.around(merged[['PMemb_x', 'PMemb_y']].to_numpy(), 2)
    probabilities = np.abs(probabilities - 0.001)

    n_bins = 10
    ax.hist2d(probabilities[:, 0], probabilities[:, 1], bins=n_bins, range=[[0., 1.], [0., 1.]], cmap='Greens',
              norm=colors.LogNorm())
    h, _, _ = np.histogram2d(probabilities[:, 0], probabilities[:, 1], bins=n_bins)
    for i in range(n_bins):
        for j in range(n_bins):
            ax.text(x=j / n_bins + 1 / n_bins / 2, y=i / n_bins + 1 / n_bins / 2, s=int(h.T[i, j]), va='center',
                    ha='center', size='x-large')

    ax.set_title(title, fontsize=20)
    ax.set_xlabel(f'Membership probability ({label2})', fontsize=18)
    ax.set_ylabel(f'Membership probability ({label1})', fontsize=18)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=16)
    
    if save:
        plt.savefig(save_file)
    if show:
        plt.show()
    plt.clf()
    plt.close('all')


def plot_venn_diagram(members1, members2, save_file='venn_diagram.png', title='Venn diagram', label1='members_1',
                      label2='members_2', show=True, save=False):
    """Plots a venn diagram that shows the overlap between the candidates and comparison/train members
    above a probability threshold of 90%, 50% and 10%.

    Args:
        members1 (Dataframe): Dataframe containing members
        members2 (Dataframe): Another dataframe containing members
        save_file (str): Path where the plot is saved
        title (Cluster): Plot title
        label1 (string): Label of the first member set
        label2 (string): Label of the second member set
        show (bool): Whether to show the plot
        save (bool): Whether to save the plot
    """

    fig, ax = plt.subplots(1, 3, figsize=(10, 6))
    fig.tight_layout(pad=0.)

    for idx, plot_prob_threshold in enumerate([0.9, 0.5, 0.1]):

        members1_subset = members1[members1['PMemb'] >= plot_prob_threshold]
        members2_subset = members2[members2['PMemb'] >= plot_prob_threshold]
        merged = pd.merge(members2_subset, members1_subset, how='inner', on='source_id')
        n_merged = len(merged)

        if n_merged != 0:
            v = venn2(subsets=(len(members2_subset) - n_merged, len(members1_subset) - n_merged, n_merged),
                      set_labels=(label2, label1), ax=ax[idx])

            for text in v.set_labels:
                text.set_fontsize(16)
            for text in v.subset_labels:
                text.set_fontsize(15)
            ax[idx].set_title(r'$p\geq$' + f'{int(plot_prob_threshold * 100)}%',
                              fontsize=18)

    fig.suptitle(title, fontsize=20)
    
    if save:
        plt.savefig(save_file)
    if show:
        plt.show()
    plt.clf()
    plt.close('all')


def member_colors(c_map, members, color_norm):
    """Determines the colors of the members based on their membership probability.

    Args:
        c_map (str): Label of a color map (e.g. 'Greens')
        members (Dataframe): Dataframe containing member sources
        color_norm (Normalize): A matplotlib.colors.Normalize object that defines the boundary color values

    """
    mapper = cm.ScalarMappable(norm=color_norm, cmap=c_map)
    color = np.array([(mapper.to_rgba(v)) for v in members['PMemb']])
    return color


def members_plot(ax, x, y, members, label='members'):
    """Plots the candidates with membership probabilities above a certain threshold against a background
    of non-members in sky position, proper motion, parallax and the colour-magnitude diagram.

    Args:
        ax (AxesSubplot): AxesSubplot object on which to make the plot
        x (str): Label of the x-axis property
        y (str): Label of the y-axis property
        members (Dataframe): Dataframe containing members
        label (string): Label of the member set

    """
    ax.scatter(members[x], members[y], label=label, s=15, rasterized=True, zorder=1)


def candidates_plot(ax, x, y, members, candidates, label='members'):
    """Plots the training members, candidates and non-members in sky position, proper motion, parallax
    and the colour-magnitude diagram.

    Args:
        ax (AxesSubplot): AxesSubplot object on which to make the plot
        x (str): Label of the x-axis property
        y (str): Label of the y-axis property
        members (Dataframe): Dataframe containing the member sources
        candidates (Dataframe): Dataframe containing the candidate sources
        label (string): Label of the member set

    """

    ax.scatter(members[x], members[y], label=label, s=15.0, c=np.ones((len(members))), cmap='winter',
               rasterized=True, zorder=2)
    ax.scatter(candidates[x], candidates[y], label=f'candidates', s=1.0, c='orange', rasterized=True, zorder=1)


def comparison_plot(ax, x, y, members1, members2, label1='members_1', label2='members_2'):
    """Plots the candidates and comparison members, if available, in sky position, proper motion, parallax
    and the colour-magnitude diagram. Optionally also plots the training members. A color map indicates the
    membership probability of the sources.

    Args:
        ax (AxesSubplot): AxesSubplot object on which to make the plot
        x (str): Label of the x-axis property
        y (str): Label of the y-axis property
        members1 (Dataframe): Dataframe containing members
        members2 (Dataframe): Another dataframe containing members
        label1 (string): Label of the first member set
        label2 (string): Label of the second member set

    """
    x_error = f'{x}_error'
    y_error = f'{y}_error'

    vmax = 1.2
    vmin = -0.3
    ax.scatter(members1[x], members1[y], label=label1, s=30.0 * members1['PMemb'], vmin=vmin, vmax=vmax,
               c=members1['PMemb'], cmap='Reds', zorder=2, alpha=0.7)
    ax.scatter(members2[x], members2[y], label=label2, s=10.0 * members2['PMemb'], vmin=vmin, vmax=vmax,
               c=members2['PMemb'], cmap='Greens', zorder=3, alpha=0.7)

    color_norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    if x != 'l':
        color = member_colors('Reds', members1, color_norm)
        ax.errorbar(members1[x], members1[y], xerr=members1[x_error], markersize=0.0, yerr=members1[y_error],
                    fmt='none', ecolor=color, elinewidth=0.1, zorder=0)
        color = member_colors('Greens', members2, color_norm)
        ax.errorbar(members2[x], members2[y], xerr=members2[x_error], markersize=0.0, yerr=members2[y_error],
                    fmt='none', ecolor=color, elinewidth=0.1, zorder=1)


def unique_members_plot(ax, x, y, members1, members2, label='unique members'):
    """Plots the members which are in members set 1 but not in members set 2.

    Args:
        ax (AxesSubplot): AxesSubplot object on which to make the plot
        x (str): Label of the x-axis property
        y (str): Label of the y-axis property
        members1 (Dataframe): Dataframe containing members
        members2 (Dataframe): Another dataframe containing members
        label (string): Label of the unique members

    """
    unique_members = members1[~members1['source_id'].isin(members2['source_id'])]

    vmax = 1.0
    vmin = 0.0
    x_error = f'{x}_error'
    y_error = f'{y}_error'

    ax.scatter(unique_members[x], unique_members[y], label=label, s=20.0, vmin=vmin,
               vmax=vmax, c=unique_members['PMemb'], cmap='autumn_r', zorder=1)
    color_norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    if x != 'l':
        color = member_colors('autumn_r', unique_members, color_norm)
        ax.errorbar(unique_members[x], unique_members[y], xerr=unique_members[x_error],
                    markersize=0.0, yerr=unique_members[y_error], fmt='none', ecolor=color, elinewidth=0.1, zorder=0)


def plot_zero_error_boundaries(ax, x, y, cluster):
    if cluster is not None:
        if cluster.delta_pm is not None:
            if x == 'pmra':
                delta_pm = cluster.delta_pm
                ellipse = Ellipse((cluster.pmra, cluster.pmdec), width=2 * delta_pm, height=2 * delta_pm,
                                  facecolor='none', edgecolor='red', ls='--', zorder=4, linewidth=1.5,
                                  label='zero-error boundary')
                ax.add_patch(ellipse)
            elif y == 'parallax':
                x_min, x_max = ax.get_xlim()
                ax.hlines((cluster.parallax - cluster.delta_plx_plus, cluster.parallax + cluster.delta_plx_min), x_min,
                          x_max, colors='red', linestyles='dashed', linewidths=1.5, zorder=4,
                          label='zero-error boundary')
            elif y == 'phot_g_mean_mag' and cluster.isochrone is not None:
                xs = np.linspace(cluster.isochrone[cluster.isochrone_colour].min() - 1.0,
                                 cluster.isochrone[cluster.isochrone_colour].max() + 1.0, 50)
                ys = np.linspace(cluster.isochrone['phot_g_mean_mag'].min() - 2.0,
                                 cluster.isochrone['phot_g_mean_mag'].max() + 2.0, 200)

                xx, yy = np.meshgrid(xs, ys)
                grid_stars = pd.DataFrame.from_dict({cluster.isochrone_colour: xx.flatten(), 
                                                     'phot_g_mean_mag': yy.flatten()})
                f_iso_f = isochrone_features_function(cluster.isochrone, cluster.delta_c, cluster.delta_g,
                                                      colour=cluster.isochrone_colour, scale_features=True)
                zz = np.linalg.norm(np.stack(grid_stars.apply(f_iso_f, axis=1).to_numpy()), axis=1).reshape(200, 50)
                c = ax.contour(xx, yy, zz, levels=[1.0], colors='red', linestyles='dashed', linewidths=1.5, zorder=4)
                c.collections[0].set_label('zero-error boundary')
        else:
            raise UserWarning('The maximum separation Delta are not defined for this cluster. Use'
                              ' the Cluster.set_feature_parameters() function to set them.')
    else:
        raise TypeError('A Cluster() object needs to be supplied to show zero error boundaries.')


def plot_features(ax, x, y, source, cluster):
    if cluster is not None:
        if cluster.delta_pm is not None:
            if x == 'pmra' or x == 'l':
                ax.annotate("", xy=(source[x], source[y]), xytext=(getattr(cluster, x), getattr(cluster, y)),
                            arrowprops=dict(arrowstyle="->", linewidth=1, color='black', mutation_scale=15))
            if y == 'parallax':
                ax.annotate("", xy=(source[x], source[y]), xytext=(source[x], getattr(cluster, y)),
                            arrowprops=dict(arrowstyle="->", linewidth=1, color='black', mutation_scale=15))
                x_min, x_max = ax.get_xlim()
                ax.hlines(cluster.parallax, x_min, x_max, colors='black', linestyles='dashed', linewidths=1.5, zorder=4,
                          label='mean parallax')
            if x == cluster.isochrone_colour and cluster.isochrone is not None:
                f_iso_f = isochrone_features_function(cluster.isochrone, cluster.delta_c, cluster.delta_g,
                                                      colour=cluster.isochrone_colour)
                vector_to_isochrone = source.apply(f_iso_f, axis=1).to_numpy()[0]
                ax.annotate("", xy=(source[x],
                                    source[y]),
                            xytext=(source[x],
                                    source[y] + vector_to_isochrone[1]),
                            arrowprops=dict(arrowstyle="->", linewidth=1, color='black', mutation_scale=15))
                ax.annotate("", xy=(source[x],
                                    source[y] + vector_to_isochrone[1]),
                            xytext=(source[x] + vector_to_isochrone[0],
                                    source[y] + vector_to_isochrone[1]),
                            arrowprops=dict(arrowstyle="->", linewidth=1, color='black', mutation_scale=15))
            ax.scatter(source[x], source[y], marker='*', s=100.0, c='red', rasterized=True, zorder=4)
        else:
            raise UserWarning('The maximum separation Delta are not defined for this cluster. Use'
                              ' the Cluster.set_feature_parameters() function to set them.')
    else:
        raise TypeError('A Cluster() object needs to be supplied to show features.')


def plot_sources_limits(sources, colour):
    """Determines plot axis limits by taking the maximum and minimum values of a set of sources

    Args:
        sources (Dataframe): Dataframe containing sources
        colour (str): Which colour field to use ('bp_rp', 'g_rp')

    Returns:
        limits (dict): Dictionary containing lists of upper and lower limits for the fields
            ('l', 'b', 'pmra', 'pmdec', 'parallax', 'phot_g_mean_mag', colour)

    """
    plot_fields = ['l', 'b', 'pmra', 'pmdec', 'parallax', 'phot_g_mean_mag', colour]
    minima = sources[plot_fields].min()
    maxima = sources[plot_fields].max()

    padding = {field: 0.05 * (maxima[field] - minima[field]) for field in plot_fields}
    limits = {field: [minima[field] - padding[field], maxima[field] + padding[field]] for field in plot_fields}
    limits[colour][0] = max(-1, limits[colour][0])
    limits[colour][1] = min(2.5, limits[colour][1])
    return limits


def plot_sources(members, save_file='members.png', colour='g_rp', comparison=None, candidates=None, field_sources=None,
                 plot_type='members', members_label='members', comparison_label='comparison', title='Cluster',
                 limits=None, show_features_source_id=None, show_isochrone=False, show_boundaries=False, cluster=None,
                 fig_size=(16., 16.), title_size=30., axis_label_size=20., tick_label_size=18., legend_label_size=14.,
                 show=True, save=False):
    """Main function for creating several plots that show the distribution of the sources in sky position,
    proper motion, parallax and the colour-magnitude diagram.

    Args:
        members (Dataframe): Dataframe containing members
        save_file (str): Path where the plot is saved
        colour (str): Which colour field to use ('bp_rp', 'g_rp')
        comparison (Dataframe): Another dataframe containing members
        candidates (Dataframe): Dataframe containing candidates
        field_sources (Dataframe): Dataframe containing field_sources
        plot_type (str): The kind of plot to make ('candidates', 'comparison', 'new_members', 'unique_members')
        members_label (str): Label for the members
        comparison_label (str): Label for the comparison members
        title (str): Plot title
        limits (dict): Optional dictionary containing lists of upper and lower limits for the fields
            ('l', 'b', 'pmra', 'pmdec', 'parallax', 'phot_g_mean_mag', cluster.isochrone_colour)
        show_features_source_id (int): Identity of a source for which to display its features with vectors
        show_isochrone (bool): Whether to plot the isochrone
        show_boundaries (bool): Whether to show zero-error boundaries
        cluster (Cluster): Cluster object
        fig_size (tuple, float): Tuple containing the dimension of the figure (width, height)
        title_size (float): Size of the plot title
        axis_label_size (float): Size of the axis labels
        tick_label_size (float): Size of the tick labels
        legend_label_size (float): Size of the legend labels
        show (bool): Whether to show the plot
        save (bool): Whether to save the plot

    """
    x_fields = ['l', 'phot_g_mean_mag', 'pmra', colour]
    y_fields = ['b', 'parallax', 'pmdec', 'phot_g_mean_mag']

    if colour == 'bp_rp':
        colour_label = r'$G_{BP} - G_{RP}$ [mag]'
    elif colour == 'g_rp':
        colour_label = r'$G - G_{RP}$ [mag]'
    else:
        colour_label = r'$C$ [mag]'

    labels = {'l': r'$l$ [deg]',
              'b': r'$b$ [deg]',
              'pmra': r'$\mu_{\alpha}$ [mas/yr]',
              'pmdec': r'$\mu_{\delta}$ [mas/yr]',
              'parallax': r'$\varpi$ [mas]',
              'phot_g_mean_mag': r'$G$ [mag]',
              colour: colour_label}

    if show_features_source_id is not None:
        all_sources = members
        for subset in [candidates, field_sources, comparison]:
            if subset is not None:
                all_sources = pd.concat((all_sources, subset))
        source_show_features = all_sources[all_sources['source_id'] == show_features_source_id]
        if len(source_show_features) == 0:
            raise ValueError(f'Can not show features of source with identity {show_features_source_id}, as it is not '
                             f'among the supplied sources!')
        if len(source_show_features) > 1:
            source_show_features = source_show_features.iloc[:1]
    else:
        source_show_features = None

    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    fig.tight_layout(pad=7.0)

    for i, (x, y) in enumerate(zip(x_fields, y_fields)):
        ax = axes[int(i / 2), i % 2]

        if plot_type == 'members':
            members_plot(ax, x, y, members, label=members_label)
        elif plot_type == 'candidates':
            candidates_plot(ax, x, y, members, candidates, label=members_label)
        elif plot_type == 'comparison':
            comparison_plot(ax, x, y, members, comparison, label1=members_label, label2=comparison_label)
        elif plot_type == 'unique_members':
            unique_members_plot(ax, x, y, members, comparison, label=members_label)
        else:
            raise ValueError(f'Unknown plot type: {plot_type}')
        if field_sources is not None:
            ax.scatter(field_sources[x], field_sources[y], label='non-members', s=1.0, c='gray', alpha=0.2,
                       rasterized=True, zorder=-2)

        if limits is not None:
            if x in limits:
                ax.set_xlim(limits[x])
            if y in limits:
                ax.set_ylim(limits[y])
        else:
            ax.set_xlim(ax.get_xlim())
            ax.set_ylim(ax.get_ylim())

        if show_boundaries:
            plot_zero_error_boundaries(ax, x, y, cluster)

        if source_show_features is not None:
            plot_features(ax, x, y, source_show_features, cluster)

        if y == 'phot_g_mean_mag':
            if show_isochrone and cluster is not None and cluster.isochrone is not None:
                ax.plot(cluster.isochrone[colour], cluster.isochrone['phot_g_mean_mag'],
                        label='isochrone', zorder=1)
            ax.invert_yaxis()
            legend_loc = 'best'
        else:
            legend_loc = 'upper left'

        ax.set_xlabel(labels[x], fontsize=axis_label_size)
        ax.set_ylabel(labels[y], fontsize=axis_label_size)
        ax.tick_params(labelsize=tick_label_size)
        ax.legend(fontsize=legend_label_size, loc=legend_loc)

    fig.suptitle(title, fontsize=title_size)
    if save:
        plt.savefig(save_file)
    if show:
        plt.show()
    plt.clf()
    plt.close('all')
