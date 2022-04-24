import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib_venn import venn2
from diagnostics import make_density_profile, make_mass_segregation_profile, fit_king_model_mcmc, \
    approximate_tidal_radius, fit_king_model
from data_filters import fields


def plot_density_profile(sources, save_dir, cluster, show=True, save=True):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for members, lab, col in zip([sources.comparison_members.hp(), sources.candidates.hp()],
                                 [f'{sources.comparison_members_ref}', 'this study'], ['gray', 'blue']):
        rs, densities, sigmas, areas = make_density_profile(members, cluster)
        ax.errorbar(rs, densities, sigmas, fmt="o", capsize=3.0, label=lab, c=col)

        r_fit, density_fit, r_t = fit_king_model(rs, densities, areas)
        r_t = approximate_tidal_radius(r_fit, density_fit)
        if lab == sources.comparison_members_ref:
            cluster.add_comparison_tidal_radius(r_t)
        else:
            cluster.add_tidal_radius(r_t)
        if r_t > 60:
            legend_r_t = '>60'
        else:
            legend_r_t = f'={np.round(r_t, 1)}'
        ax.plot(r_fit, density_fit, label=f'{lab}' + r', King model fit ($R_{t}$' + f'{legend_r_t} pc)', c=col)

    ax.set_yscale('log')
    ax.set_title(f'{cluster.name}'.replace('_', ' '), fontsize=18)
    ax.set_xlabel('r [pc]', fontsize=16)
    ax.set_ylabel(r'$\rho$ [stars / pc$^{2}$]', fontsize=16)
    ax.legend()
    if save:
        plt.savefig(os.path.join(save_dir, f"density_profile.png"))
    if show:
        plt.show()
    plt.close(fig)


def plot_mass_segregation_profile(sources, save_dir, cluster, show=True, save=True):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    bins, ms, sigmas = make_mass_segregation_profile(sources.comparison_members.hp(tidal_radius=cluster.r_t_c), cluster)
    ax.errorbar(bins, ms, sigmas, elinewidth=0.5, ms=2.0, capsize=2.0, fmt="o", c='gray',
                label=f'{sources.comparison_members_ref}')

    bins, ms, sigmas = make_mass_segregation_profile(sources.candidates.hp(tidal_radius=cluster.r_t), cluster)
    ax.errorbar(bins, ms, sigmas, elinewidth=0.5, ms=2.0, capsize=2.0, fmt="o", c='black', label='this study')

    ax.hlines(1, np.min(bins), np.max(bins), linestyles='dashed')
    ax.set_title(f'{cluster.name}'.replace('_', ' '), fontsize=18)
    ax.set_xlabel(r'$N_{MST}$', fontsize=16)
    ax.set_ylabel(r'$\Lambda_{MSR}$', fontsize=16)
    ax.legend()
    if save:
        plt.savefig(os.path.join(save_dir, f"mass_segregation.png"))
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix(sources, cluster, save_dir, show=True, save=True):
    new_members = sources.candidates.hp(0, cluster.r_t)
    comparison = sources.comparison_members.hp(0, cluster.r_t_c)

    merged = pd.merge(comparison, new_members, how='outer', on='source_id')
    probs = np.round(merged[['PMemb_x', 'PMemb_y']].to_numpy(), 2)
    probs[np.isnan(probs)] = 0.0
    probs += 0.00001
    probs = np.where(probs > 1.0, 1.0, probs)
    n_bins = 10
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.hist2d(probs[:, 0], probs[:, 1], bins=n_bins, range=[[0., 1.], [0., 1.]], cmap='Greens',
              norm=colors.LogNorm())
    h, _, _ = np.histogram2d(probs[:, 0], probs[:, 1], bins=n_bins)
    for i in range(n_bins):
        for j in range(n_bins):
            ax.text(x=j / n_bins + 1 / n_bins / 2, y=i / n_bins + 1 / n_bins / 2, s=int(h.T[i, j]), va='center',
                    ha='center', size='x-large')
    ax.set_xlabel(f'Membership probability ({sources.comparison_members_ref})', fontsize=16)
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


def plot_sources(sources, cluster, save_dir, isochrone_df=None, plot_type='candidates', show_probs=True, show=True,
                 save=True):

    x_fields = ['l', 'phot_g_mean_mag', 'pmra', 'bp_rp']
    y_fields = ['b', 'parallax', 'pmdec', 'phot_g_mean_mag']

    labels = {'l': r'$l$ [deg]',
              'b': r'$b$ [deg]',
              'pmra': r'$\mu_{\alpha}$ [mas/yr]',
              'pmdec': r'$\mu_{\delta}$ [mas/yr]',
              'parallax': r'$\varpi$ [mas]',
              'phot_g_mean_mag': r'$G$ [mag]',
              'bp_rp': r'$BP - RP$ [mag]'}

    minima = sources.candidates.min()
    maxima = sources.candidates.max()

    frames = {field: 0.05 * (maxima[field] - minima[field]) for field in fields['plot']}
    lims = {field: [minima[field] - frames[field], maxima[field] + frames[field]] for field in fields['plot']}

    fig, axes = plt.subplots(2, 2, figsize=(18, 18))
    fig.tight_layout(pad=5.0)

    for i, (x, y) in enumerate(zip(x_fields, y_fields)):
        ax = axes[int(i / 2), i % 2]

        if plot_type == 'candidates':
            candidates_plot(ax, x, y, sources, show_probs)
        elif plot_type == 'comparison':
            comparison_plot(ax, x, y, sources, cluster)
        elif plot_type == 'members':
            members_plot(ax, x, y, sources, cluster)

        ax.set_xlim(lims[x])
        ax.set_ylim(lims[y])
        if y == 'phot_g_mean_mag':
            if isochrone_df is not None and plot_type != 'members':
                ax.plot(isochrone_df['bp_rp'], isochrone_df['phot_g_mean_mag'], label='isochrone', zorder=-1)
            ax.invert_yaxis()
        ax.set_xlabel(labels[x], fontsize=16)
        ax.set_ylabel(labels[y], fontsize=16)
        ax.set_title(f'{cluster.name}'.replace('_', ' '), fontsize=18)
        ax.legend(loc='lower left')
    if save:
        plt.savefig(os.path.join(save_dir, f"{plot_type}.png"))
    if show:
        plt.show()
    plt.close(fig)


def plot_venn_diagram(sources, cluster, save_dir, show, save):
    new_members = sources.candidates.hp(tidal_radius=cluster.r_t)
    comparison = sources.comparison_members.hp(tidal_radius=cluster.r_t_c)
    merged = pd.merge(comparison, new_members, how='inner', on='source_id')

    n_merged = len(merged)

    venn2(subsets=(len(comparison) - n_merged, len(new_members) - n_merged, n_merged),
          set_labels=(f'{sources.comparison_members_ref}', 'this study'))
    if save:
        plt.savefig(os.path.join(save_dir, f"venn_diagram.png"))
    if show:
        plt.show()
    plt.close()


def candidates_plot(ax, x, y, sources, show_probs):
    train_members = sources.train_members.hp()
    candidates = sources.candidates
    noise = sources.noise

    ax.scatter(train_members[x], train_members[y],
               label=f'train members (p>={sources.prob_threshold}) ({sources.train_members_ref})', s=8.0,
               c=np.ones((len(train_members))), cmap='winter', rasterized=True, zorder=1)
    if show_probs:
        ax.scatter(candidates[x], candidates[y], label=f'candidates', s=4.0,
                   c=candidates['PMemb'], rasterized=True, cmap='autumn_r', zorder=0)
    else:
        ax.scatter(candidates[x], candidates[y], label=f'candidates', s=1.0, c='orange',
                   rasterized=True, zorder=0)
    ax.scatter(noise[x], noise[y], label='non members', s=0.5, c='gray', alpha=0.2, rasterized=True, zorder=-2)


def comparison_plot(ax, x, y, sources, cluster):
    # train_members = sources.train_members.hp()
    comparison_members = sources.comparison_members.hp(tidal_radius=cluster.r_t_c)
    new_members = sources.candidates.hp(tidal_radius=cluster.r_t)

    if x == 'l':
        x_error = 'ra_error'
        y_error = 'dec_error'
    else:
        x_error = f'{x}_error'
        y_error = f'{y}_error'

    # ax.errorbar(train_members[x], train_members[y], xerr=train_members[f'{x}_error'],
    #             yerr=train_members[f'{y}_error'], fmt='.',
    #             label=f'train members (p>={sources.prob_threshold}) ({sources.train_members_ref})',
    #             markersize=5.0, c='blue', elinewidth=0.2, zorder=1)
    ax.errorbar(comparison_members[x], comparison_members[y], xerr=comparison_members[x_error],
                yerr=comparison_members[y_error], fmt='.',
                label=f'members (p>={sources.prob_threshold}) obtained by {sources.comparison_members_ref} ',
                markersize=5.0, c='green', elinewidth=0.2, zorder=2)
    ax.errorbar(new_members[x], new_members[y], xerr=new_members[x_error], yerr=new_members[y_error],
                fmt='.', label=f'members (p>={sources.prob_threshold}) obtained in this study', markersize=10.0, c='red', elinewidth=0.2,
                zorder=0)


def members_plot(ax, x, y, sources, cluster):
    new_members = sources.candidates.hp(tidal_radius=cluster.r_t)
    field = pd.concat((sources.noise, sources.candidates.lp(tidal_radius=cluster.r_t)))

    ax.scatter(new_members[x], new_members[y],
               label=f'members (p>={sources.prob_threshold}) obtained in this study', s=20.0,
               rasterized=True, zorder=0)
    ax.scatter(field[x], field[y], label='non members', s=1.0, c='gray', alpha=0.2, rasterized=True, zorder=-1)


def make_plots(sources, cluster, save_dir, isochrone, show, save_plots):
    plot_sources(sources, cluster, save_dir, isochrone_df=isochrone, plot_type='candidates', show=show, save=save_plots)
    print('Created candidates plot')
    plot_density_profile(sources, save_dir, cluster, show=show, save=save_plots)
    print('Created density profile plot')
    plot_mass_segregation_profile(sources, save_dir, cluster, show=show, save=save_plots)
    print('Created mass segregation plot')
    plot_confusion_matrix(sources, cluster, save_dir, show=show, save=save_plots)
    print('Created confusion matrix plot')
    plot_venn_diagram(sources, cluster, save_dir, show=show, save=save_plots)
    print('Created venn diagram plot')
    plot_sources(sources, cluster, save_dir, isochrone_df=isochrone, plot_type='members', show=show, save=save_plots)
    print('Created members plot')
    plot_sources(sources, cluster, save_dir, isochrone_df=isochrone, plot_type='comparison', show=show, save=save_plots)
    print('Created comparison plot')
