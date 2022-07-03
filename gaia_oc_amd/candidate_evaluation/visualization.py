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
from gaia_oc_amd.data_preparation.utils import norm


def plot_loss_accuracy(metrics, save_dir, show=True, save=True):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    epochs = np.arange(len(metrics['train_loss']))

    ax[0].plot(epochs, metrics['train_loss'], label='train')
    ax[0].plot(epochs, metrics['val_loss'], label='validation')

    ax[1].plot(epochs, metrics['train_pos_acc'], label='train member accuracy', c='tab:blue')
    ax[1].plot(epochs, metrics['train_neg_acc'], '--', label='train non member accuracy', c='tab:blue')
    ax[1].plot(epochs, metrics['val_pos_acc'], label='validation member accuracy', c='tab:orange')
    ax[1].plot(epochs, metrics['val_neg_acc'], '--', label='validation non member accuracy', c='tab:orange')
    
    labels = ['loss', 'accuracy (%)']
    titles = ['Loss', 'Accuracy']
    
    for i in range(2):
        ax[i].set_xticks(epochs[::5])
        ax[i].set_xlabel('epoch', fontsize=18)
        ax[i].set_ylabel(labels[i], fontsize=18)
        ax[i].set_title(titles[i], fontsize=20)
        ax[i].tick_params(labelsize=16)
        ax[i].grid(True)
        ax[i].legend(fontsize=15)
    ax[1].set_ylim(90, 101)
    
    if save:
        plt.savefig(os.path.join(save_dir, 'loss_accuracy.png'))
    if show:
        plt.show()
    plt.close(fig)


def drop_training_members(members, new_members, comparison):
    new_members = pd.merge(new_members, members, how='outer', on='source_id', suffixes=('', '_'),
                           indicator=True).query("_merge == 'left_only'").drop('_merge',
                                                                               axis=1).reset_index(drop=True)
    comparison = pd.merge(comparison, members, how='outer', on='source_id', suffixes=('', '_'),
                          indicator=True).query("_merge == 'left_only'").drop('_merge',
                                                                              axis=1).reset_index(drop=True)
    return new_members, comparison


def select_comparison(sources, prob_threshold=0.1, show_train_members=False, drop_train_members=False,
                      tidal_radius=None):
    new_members = sources.candidates.hp(min_prob=prob_threshold, tidal_radius=tidal_radius)

    if show_train_members or sources.comparison_members is None:
        comparison = sources.members
        comparison_label = sources.members_label
    else:
        comparison = sources.comparison_members
        comparison_label = sources.comparison_label

        if drop_train_members:
            members = sources.members
            new_members, comparison = drop_training_members(members, new_members, comparison)

    return new_members, comparison, comparison_label


def plot_density_profile(sources, cluster, save_dir, show_train_members=False, show=True, save=True):
    new_members, comparison, comparison_label = select_comparison(sources, show_train_members=show_train_members)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for mem, lab, col in zip([comparison, new_members], [comparison_label, 'this study'], ['gray', 'black']):
        rs, densities, sigmas, areas = make_density_profile(mem, cluster)
        r_fit, density_fit, r_t = fit_king_model(rs, densities, areas)
        ax.errorbar(rs, densities, sigmas, fmt="o", capsize=3.0, label=f'{lab}', c=col)
        ax.plot(r_fit, density_fit, label=r'King model fit, $R_{t}$' + f'={np.round(r_t, 1)} pc ' + f'({lab})', c=col)

    ax.set_yscale('log')
    ax.set_title(f'{cluster.name}'.replace('_', ' '), fontsize=20)
    ax.set_xlabel('r [pc]', fontsize=18)
    ax.set_ylabel(r'$\rho$ [stars / pc$^{2}$]', fontsize=18)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=15)
    
    if save:
        plt.savefig(os.path.join(save_dir, f"density_profile.png"))
    if show:
        plt.show()
    plt.close(fig)


def plot_mass_segregation_profile(sources, cluster, save_dir, min_n_mst=5, max_n_mst=100, n_samples=20,
                                  show_train_members=False, tidal_radius=None, show=True, save=True):
    new_members, comparison, comparison_label = select_comparison(sources, show_train_members=show_train_members,
                                                                  tidal_radius=tidal_radius)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for mem, lab, col in zip([comparison, new_members], [comparison_label, 'this study'], ['gray', 'black']):
        bins = np.arange(min_n_mst, min(len(mem), max_n_mst))
        ms, sigmas = make_mass_segregation_profile(mem, cluster, bins, n_samples)
        ax.errorbar(bins, ms, sigmas, elinewidth=0.5, ms=2.0, capsize=2.0, fmt="o", c=col, label=lab)

    ax.hlines(1, min_n_mst, max_n_mst, linestyles='dashed')
    ax.set_title(f'{cluster.name}'.replace('_', ' '), fontsize=20)
    ax.set_xlabel(r'$N_{MST}$', fontsize=18)
    ax.set_ylabel(r'$\Lambda_{MSR}$', fontsize=18)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=15)
    
    if save:
        plt.savefig(os.path.join(save_dir, f"mass_segregation.png"))
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix(sources, cluster, save_dir, tidal_radius=None, drop_train_members=False,
                          show_train_members=False, show=True, save=True):
    new_members, comparison, comparison_label = select_comparison(sources, prob_threshold=0.0,
                                                                  show_train_members=show_train_members,
                                                                  drop_train_members=drop_train_members,
                                                                  tidal_radius=tidal_radius)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    merged = pd.merge(comparison, new_members, how='outer', on='source_id').fillna(0)
    probabilities = np.around(merged[['PMemb_x', 'PMemb_y']].to_numpy(), 3)

    n_bins = 10
    ax.hist2d(probabilities[:, 0], probabilities[:, 1], bins=n_bins, range=[[0., 1.], [0., 1.]], cmap='Greens',
              norm=colors.LogNorm())
    h, _, _ = np.histogram2d(probabilities[:, 0], probabilities[:, 1], bins=n_bins)
    for i in range(n_bins):
        for j in range(n_bins):
            ax.text(x=j / n_bins + 1 / n_bins / 2, y=i / n_bins + 1 / n_bins / 2, s=int(h.T[i, j]), va='center',
                    ha='center', size='x-large')
    
    ax.set_xlabel(f'Membership probability ({comparison_label})', fontsize=16)
    ax.set_ylabel('Membership probability (this study)', fontsize=16)
    ax.set_title(f'{cluster.name}'.replace('_', ' '), fontsize=18)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=16)
    
    if save:
        plt.savefig(os.path.join(save_dir, f"membership_comparison.png"))
    if show:
        plt.show()
    plt.close(fig)


def plot_venn_diagram(sources, cluster, save_dir, tidal_radius=None, drop_train_members=False,
                      show_train_members=False, show=True, save=True):
    new_members, comparison, comparison_label = select_comparison(sources, show_train_members=show_train_members,
                                                                  drop_train_members=drop_train_members,
                                                                  tidal_radius=tidal_radius)

    fig, ax = plt.subplots(1, 3, figsize=(18, 8))

    for idx, plot_prob_threshold in enumerate([0.9, 0.5, 0.1]):

        new_members_subset = new_members[new_members['PMemb'] >= plot_prob_threshold]
        comparison_subset = comparison[comparison['PMemb'] >= plot_prob_threshold]
        merged = pd.merge(comparison_subset, new_members_subset, how='inner', on='source_id')
        n_merged = len(merged)

        v = venn2(subsets=(len(comparison_subset) - n_merged, len(new_members_subset) - n_merged, n_merged),
                  set_labels=(comparison_label, 'this study'), ax=ax[idx])

        for text in v.set_labels:
            text.set_fontsize(18)
        for text in v.subset_labels:
            text.set_fontsize(16)
        ax[idx].set_title(f'{cluster.name} '.replace('_', ' ') + r'($\geq$' + f'{int(plot_prob_threshold * 100)}%)',
                          fontsize=20)
    
    if save:
        plt.savefig(os.path.join(save_dir, f"venn_diagram.png"))
    if show:
        plt.show()
    plt.close()


def candidates_plot(ax, x, y, sources, cluster, tidal_radius=None, show_features=False, show_boundaries=True):
    members = sources.members
    candidates = sources.candidates.hp(tidal_radius=tidal_radius)
    non_members = sources.non_members

    ax.scatter(members[x], members[y], label=f'train members', s=12.0, c=np.ones((len(members))), cmap='winter',
               rasterized=True, zorder=2)
    ax.scatter(candidates[x], candidates[y], label=f'candidates', s=1.0, c='orange',
               rasterized=True, zorder=0)
    ax.scatter(non_members[x], non_members[y], label='non members', s=0.5, c='gray', alpha=0.2, rasterized=True,
               zorder=-2)

    if show_features:
        idx = np.random.randint(low=0, high=len(candidates))
        if x == 'pmra' or x == 'l':
            ax.annotate("", xy=(candidates[x].iloc[idx], candidates[y].iloc[idx]),
                        xytext=(getattr(cluster, x), getattr(cluster, y)),
                        arrowprops=dict(arrowstyle="->", linewidth=3, color='black', mutation_scale=30))
        if x == 'phot_g_mean_mag':
            ax.annotate("", xy=(candidates[x].iloc[idx], candidates[y].iloc[idx]),
                        xytext=(candidates[x].iloc[idx], getattr(cluster, y)),
                        arrowprops=dict(arrowstyle="->", linewidth=3, color='black', mutation_scale=30))
            ax.hlines(cluster.parallax, 6, 22,
                      colors='black', linestyles='dashed', linewidths=1.5, zorder=2, label='mean parallax')
        if x == 'bp_rp':
            ax.annotate("", xy=(candidates[x].iloc[idx], candidates[y].iloc[idx]),
                        xytext=(candidates[x].iloc[idx], candidates[y].iloc[idx] +
                                candidates['f_g'].iloc[idx] * 1.5),
                        arrowprops=dict(arrowstyle="->", linewidth=3, color='black', mutation_scale=30))
            ax.annotate("", xy=(candidates[x].iloc[idx], candidates[y].iloc[idx] +
                                candidates['f_g'].iloc[idx] * 1.5),
                        xytext=(candidates[x].iloc[idx] + candidates['f_c'].iloc[idx] * 0.5,
                                candidates[y].iloc[idx] + candidates['f_g'].iloc[idx] * 1.5),
                        arrowprops=dict(arrowstyle="->", linewidth=3, color='black', mutation_scale=30))
        ax.scatter(candidates[x].iloc[idx], candidates[y].iloc[idx], marker='*', s=100.0, c='red', rasterized=True,
                   zorder=3)

    if x == 'pmra' and show_boundaries:
        ellipse = Ellipse((cluster.pmra, cluster.pmdec), width=2 * cluster.pmra_delta, height=2 * cluster.pmdec_delta,
                          facecolor='none', edgecolor='red', ls='--', zorder=2, linewidth=1.5,
                          label='zero-error boundary')
        ax.add_patch(ellipse)
    elif y == 'parallax' and show_boundaries:
        ax.hlines((cluster.parallax - cluster.plx_delta_plus, cluster.parallax + cluster.plx_delta_min), 6, 22,
                  colors='red', linestyles='dashed', linewidths=1.5, zorder=2, label='zero-error boundary')


def color_map(c_map, members, color_norm):
    mapper = cm.ScalarMappable(norm=color_norm, cmap=c_map)
    color = np.array([(mapper.to_rgba(v)) for v in members['PMemb']])
    return color


def comparison_plot(ax, x, y, sources, tidal_radius=False, show_train_members=False):
    new_members = sources.candidates.hp(min_prob=0.1, tidal_radius=tidal_radius)
    comparison = sources.comparison_members
    members = sources.members

    x_error = f'{x}_error'
    y_error = f'{y}_error'

    vmax = 1.2
    vmin = -0.3

    color_norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    if show_train_members:
        ax.scatter(members[x], members[y], label=f'members ({sources.members_label})',
                   s=5.0, vmin=vmin, vmax=vmax, c=members['PMemb'], cmap='Blues', zorder=2)
    if comparison is not None:
        ax.scatter(comparison[x], comparison[y], label=f'members ({sources.comparison_label})',
                   s=10.0, vmin=vmin, vmax=vmax, c=comparison['PMemb'], cmap='Greens', zorder=2)
    ax.scatter(new_members[x], new_members[y], label=f'members (this study)',
               s=30.0, vmin=vmin, vmax=vmax, c=new_members['PMemb'], cmap='Reds', zorder=1)

    if x != 'l':
        if show_train_members:
            color = color_map('Blues', members, color_norm)
            ax.errorbar(members[x], members[y], xerr=members[x_error],
                        markersize=0.0, yerr=members[y_error], fmt='none', ecolor=color, elinewidth=0.2,
                        zorder=0)
        if comparison is not None:
            color = color_map('Greens', comparison, color_norm)
            ax.errorbar(comparison[x], comparison[y], xerr=comparison[x_error],
                        markersize=0.0, yerr=comparison[y_error], fmt='none', ecolor=color, elinewidth=0.2,
                        zorder=0)
        color = color_map('Reds', new_members, color_norm)
        ax.errorbar(new_members[x], new_members[y], xerr=new_members[x_error],
                    markersize=0.0, yerr=new_members[y_error], fmt='none', ecolor=color, elinewidth=0.2, zorder=-1)


def members_plot(ax, x, y, sources, plot_prob_threshold, tidal_radius=None):
    if tidal_radius:
        new_members = sources.candidates.hp(min_prob=plot_prob_threshold, tidal_radius=tidal_radius)
        non_members = pd.concat((sources.non_members, sources.candidates.lp(max_prob=plot_prob_threshold,
                                                                            tidal_radius=tidal_radius)))
    else:
        new_members = sources.candidates.hp(min_prob=plot_prob_threshold)
        non_members = pd.concat((sources.non_members, sources.candidates.lp(max_prob=plot_prob_threshold)))

    ax.scatter(new_members[x], new_members[y], label=r'members ($p \geq$' + f'{plot_prob_threshold})', s=20.0,
               rasterized=True, zorder=0)
    ax.scatter(non_members[x], non_members[y], label='non members', s=1.0, c='gray', alpha=0.2, rasterized=True,
               zorder=-1)


def plot_sources(sources, cluster, save_dir, isochrone=None, plot_type='candidates', prob_threshold=1.0,
                 tidal_radius=None, show_features=False, show_boundaries=True, show_train_members=False,
                 show=True, save=True):
    x_fields = ['l', 'phot_g_mean_mag', 'pmra', 'bp_rp']
    y_fields = ['b', 'parallax', 'pmdec', 'phot_g_mean_mag']
    plot_fields = ['l', 'b', 'pmra', 'pmdec', 'parallax', 'phot_g_mean_mag', 'bp_rp']

    labels = {'l': r'$l$ [deg]',
              'b': r'$b$ [deg]',
              'pmra': r'$\mu_{\alpha}$ [mas/yr]',
              'pmdec': r'$\mu_{\delta}$ [mas/yr]',
              'parallax': r'$\varpi$ [mas]',
              'phot_g_mean_mag': r'$G$ [mag]',
              'bp_rp': r'$G_{BP} - G_{RP}$ [mag]'}

    minima = sources.all_sources[plot_fields].min()
    maxima = sources.all_sources[plot_fields].max()

    padding = {field: 0.05 * (maxima[field] - minima[field]) for field in plot_fields}
    limits = {field: [minima[field] - padding[field], maxima[field] + padding[field]] for field in plot_fields}

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.tight_layout(pad=5.0)

    for i, (x, y) in enumerate(zip(x_fields, y_fields)):
        ax = axes[int(i / 2), i % 2]

        if plot_type == 'candidates':
            candidates_plot(ax, x, y, sources, cluster, tidal_radius, show_features, show_boundaries)
        elif plot_type == 'comparison':
            comparison_plot(ax, x, y, sources, tidal_radius, show_train_members)
        elif plot_type == 'new_members':
            members_plot(ax, x, y, sources, prob_threshold, tidal_radius)

        ax.set_xlim(limits[x])
        ax.set_ylim(limits[y])

        if y == 'phot_g_mean_mag':
            ax.set_xlim(max(-1, limits[x][0]), limits[x][1])
            if isochrone is not None:
                ax.plot(isochrone['bp_rp'], isochrone['phot_g_mean_mag'], label='isochrone', zorder=1)
                if plot_type == 'candidates' and not show_features:
                    xs = np.linspace(-1., 4., 51)
                    ys = np.linspace(0, 22, 221)

                    xx, yy = np.meshgrid(xs, ys)
                    grid_stars = pd.DataFrame.from_dict({'bp_rp': xx.flatten(), 'phot_g_mean_mag': yy.flatten()})
                    zz = norm(np.stack(grid_stars.apply(isochrone_features_function(cluster, isochrone),
                                                        axis=1).to_numpy()), axis=1).reshape(221, 51)
                    c = ax.contour(xx, yy, zz, levels=[1.0], colors='red', linestyles='dashed', linewidths=1.5)
                    c.collections[0].set_label('zero-error boundary')
            ax.invert_yaxis()

        ax.set_xlabel(labels[x], fontsize=18)
        ax.set_ylabel(labels[y], fontsize=18)
        ax.tick_params(labelsize=16)
        ax.set_title(f'{cluster.name}'.replace('_', ' '), fontsize=20)
        ax.legend(fontsize=15)
    
    if save:
        plt.savefig(os.path.join(save_dir, f"{plot_type}.png"))
    if show:
        plt.show()
    plt.close(fig)


def make_plots(sources, cluster, save_dir, isochrone, prob_threshold=1.0, show=True, save=True,
               tidal_radius=None, show_features=False, show_boundaries=True, show_train_members=False):
    plot_density_profile(sources, cluster, save_dir, show=show, save=save)
    plot_mass_segregation_profile(sources, cluster, save_dir, tidal_radius=tidal_radius, show=show, save=save)
    plot_confusion_matrix(sources, cluster, save_dir, tidal_radius=tidal_radius, show=show, save=save)
    plot_venn_diagram(sources, cluster, save_dir, tidal_radius=tidal_radius, show=show, save=save)

    plot_sources(sources, cluster, save_dir, isochrone=isochrone, plot_type='new_members',
                 prob_threshold=prob_threshold, tidal_radius=tidal_radius, show_features=show_features,
                 show_boundaries=show_boundaries, show_train_members=show_train_members, show=show, save=save)
    plot_sources(sources, cluster, save_dir, isochrone=isochrone, plot_type='comparison',
                 prob_threshold=prob_threshold, tidal_radius=tidal_radius, show_features=show_features,
                 show_boundaries=show_boundaries, show_train_members=show_train_members, show=show, save=save)
