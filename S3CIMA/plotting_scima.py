"""
last edit on The Nov  3 19:12:31 2022

@author: sepidehbabaei
"""
""" Copyright 2016-2017 ETH Zurich, Eirini Arvaniti and Manfred Claassen.

This module contains functions for plotting the results of a CellCnn analysis.

"""



import os
import sys
import logging
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from utils import mkdir_p
import statsmodels.api as sm
from collections import Counter

logger = logging.getLogger(__name__)
import numpy.matlib
from sklearn.preprocessing import StandardScaler


  

def plot_results(results, samples, phenotypes, labels, outdir, sample_pat_id,
                 filter_diff_thres=.2, filter_response_thres=0, response_grad_cutoff=None,
                 stat_test=None, log_yscale=False,
                 group_a='group A', group_b='group B', group_names=None,
                 regression=False, show_filters=True, ALL = False):
    """ Plots the results of a CellCnn analysis.

    Args:
        - results :
            Dictionary containing the results of a CellCnn analysis.
        - samples :
            Samples from which to visualize the selected cell populations.
        - phenotypes :
            List of phenotypes corresponding to the provided `samples`.
        - labels :
            Names of measured markers.
        - outdir :
            Output directory where the generated plots will be stored.
        - filter_diff_thres :
            Threshold that defines which filters are most discriminative. Given an array
            ``filter_diff`` of average cell filter response differences between classes,
            sorted in decreasing order, keep a filter ``i, i > 0`` if it holds that
            ``filter_diff[i-1] - filter_diff[i] < filter_diff_thres * filter_diff[i-1]``.
            For regression problems, the array ``filter_diff`` contains Kendall's tau
            values for each filter.
        - response_grad_cutoff :
            Threshold on the gradient of the cell filter response CDF, might be useful for defining
            the selected cell population.
        - stat_test: None | 'ttest' | 'mannwhitneyu'
            Optionally, perform a statistical test on selected cell population frequencies between
            two groups and report the corresponding p-value on the boxplot figure
            (see plots description below). Default is None. Currently only used for binary
            classification problems.
        - group_a :
            Name of the first class.
        - group_b :
            Name of the second class.
        - group_names :
            List of names for the different phenotype classes.
        - log_yscale :
            If True, display the y-axis of the boxplot figure (see plots description below) in
            logarithmic scale.
        - umap_ncell :
            Number of cells to include in UMAP calculations and plots.
        - regression :
            Whether it is a regression problem.
        - show_filters :
            Whether to plot learned filter weights.

    Returns:
        A list with the indices and corresponding cell filter response thresholds of selected
        discriminative filters. \
        This function also produces a collection of plots for model interpretation.
        These plots are stored in `outdir`. They comprise the following:

        - clustered_filter_weights.pdf :
            Filter weight vectors from all trained networks that pass a validation accuracy
            threshold, grouped in clusters via hierarchical clustering. Each row corresponds to
            a filter. The last column(s) indicate the weight(s) connecting each filter to the output
            class(es). Indices on the y-axis indicate the filter cluster memberships, as a
            result of the hierarchical clustering procedure.
        - consensus_filter_weights.pdf :
            One representative filter per cluster is chosen (the filter with minimum distance to all
            other memebers of the cluster). We call these selected filters "consensus filters".
        - best_net_weights.pdf :
            Filter weight vectors of the network that achieved the highest validation accuracy.
        - filter_response_differences.pdf :
            Difference in cell filter response between classes for each consensus filter.
            To compute this difference for a filter, we first choose a filter-specific class, that's
            the class with highest output weight connection to the filter. Then we compute the
            average cell filter response (value after the pooling layer) for validation samples
            belonging to the filter-specific class (``v1``) and the average cell filter response
            for validation samples not belonging to the filter-specific class (``v0``).
            The difference is computed as ``v1 - v0``. For regression problems, we cannot compute
            a difference between classes. Instead we compute Kendall's rank correlation coefficient
            between the predictions of each individual filter (value after the pooling layer) and
            the true response values.
            This plot helps decide on a cutoff (``filter_diff_thres`` parameter) for selecting
            discriminative filters.
        - umap_all_cells.png :
            Marker distribution overlaid on UMAP map. 

        In addition, the following plots are produced for each selected filter (e.g. filter ``i``):

        - cdf_filter_i.pdf :
            Cumulative distribution function of cell filter response for filter ``i``. This plot
            helps decide on a cutoff (``filter_response_thres`` parameter) for selecting the
            responding cell population.

        - selected_population_distribution_filter_i.pdf :
            Histograms of univariate marker expression profiles for the cell population selected by
            filter ``i`` vs all cells.

        - selected_population_frequencies_filter_i.pdf :
            Boxplot of selected cell population frequencies in samples of the different classes,
            if running a classification problem. For regression settings, a scatter plot of selected
            cell population frequencies vs response variable is generated.

        - umap_cell_response_filter_i.png :
            Cell filter response overlaid on UMAP map.

        - umap_selected_cells_filter_i.png :
            Marker distribution of selected cell population overlaid on UMAP map.
            :rtype: object
    """
    # create the output directory
    mkdir_p(outdir)

    # number of measured markers
    #nmark = samples[0].shape[1]
    nmark = len(labels)
    print('number of measured markers', nmark)
    if results['selected_filters'] is not None:
        logger.info("Loading the weights of consensus filters.")
        filters = results['selected_filters']
    else:
        sys.exit('Consensus filters were not found.')

    if show_filters:
        plot_filters(results, labels, outdir)
    # get discriminative filter indices in consensus matrix
    #keep_idx = np.arange(results['selected_filters'].shape[0])
    keep_idx = discriminative_filters(results, outdir, filter_diff_thres,show_filters=show_filters)
    print('keep_idx', keep_idx)
    # encode the sample and sample-phenotype for each cell
    sample_sizes = []
    per_cell_ids = []
    if ALL:
        for i, x in enumerate(samples):
            sample_sizes.append(1)                # sample size that is fed to the plot routine
            per_cell_ids.append(i)   # every cells gets an id corresponding to the sample, cells belong to sample 0 gets and id 0
    else:   
        for i, x in enumerate(samples):
            sample_sizes.append(x.shape[0])                # sample size that is fed to the plot routine
            per_cell_ids.append(i * np.ones(x.shape[0]))   
   
    x = np.vstack(samples)
    z = np.hstack(per_cell_ids)

    print('sample size',x.shape[0])
    print('patient id list', Counter(sample_pat_id)) 
    
    scaler = StandardScaler()
    x = scaler.fit_transform(x)


    # compute vmin and vmax for each columns (markers) of the sample data
    vmin, vmax = np.zeros(x.shape[1]), np.zeros(x.shape[1])
    for seq_index in range(x.shape[1]):
        vmin[seq_index] = np.percentile(x[:, seq_index], 1)
        vmax[seq_index] = np.percentile(x[:, seq_index], 99)


    return_filters = []
    for i_filter in keep_idx:
        w = filters[i_filter, :nmark]
        b = filters[i_filter, nmark]
        g = np.sum(w.reshape(1, -1) * x, axis=1) + b
                     
        ecdf = sm.distributions.ECDF(g)         # calculate the cumulative pdf
        gx = np.linspace(np.min(g), np.max(g))  # default num is 50
        gy = ecdf(gx)
        
        t= np.quantile(g, 0.90) #mquantiles(g, prob=[0.1, 0.2, 0.5, 0.8],)[0]
        print('thresh:', t)
        plt.figure()
        sns.set_style('whitegrid')
        a = plt.step(gx, gy)
        if response_grad_cutoff is not None:
            by = np.array(a[0].get_ydata())[::-1]
            bx = np.array(a[0].get_xdata())[::-1]
            b_diff_idx = np.where(by[:-1] - by[1:] >= response_grad_cutoff)[0]
            if len(b_diff_idx) > 0:
                t = bx[b_diff_idx[0] + 1]
        plt.plot((t, t), (np.min(gy), 1.), 'r--')
        plt.xlabel('Cell filter response')
        plt.ylabel('Cumulative distribution function (CDF)')
        sns.despine()
        plt.savefig(os.path.join(outdir, 'cdf_filter_%d.pdf' % i_filter), format='pdf')
        plt.clf()
        plt.close()

        condition = g > t
        #condv = np.vstack(condition)
        condg = np.vstack(g)
        #import pickle
        #pickle.dump(condg, open(os.path.join(outdir, 'g_%d.p' % i_filter),'wb'))
        #np.savetxt(os.path.join(outdir, 'condition_%d.txt' % i_filter), condv,delimiter=" ", fmt="%s")
        np.savetxt(os.path.join(outdir, 'g_%d.txt' % i_filter), condg,delimiter=" ", fmt="%s")

        z1 = z[condition]     # select the per-cell ids based on the chosen cut-off
        x1 = x[condition]     # select the  cells  based on the chosen cut-off
        
        # skip a filter if it does not select any cell with the new cutoff threshold
        if x1.shape[0] == 0:
            print('skipping a filter')
            continue
        suffix = str(np.around(t,2))+'_'+'filter_%d' % i_filter
        if not ALL:
            plot_selected_subset(x1, z1, x, labels, sample_sizes, phenotypes,
                            outdir, suffix, stat_test, log_yscale,
                            group_a, group_b, group_names, regression)

                   
        # else add the filters to selected filters
        return_filters.append((i_filter, t))

    logger.info("Done.")
    return return_filters



def discriminative_filters(results, outdir, filter_diff_thres, show_filters=True):
    mkdir_p(outdir)
    keep_idx = np.arange(results['selected_filters'].shape[0])

    # select the discriminative filters based on the validation set
    if 'filter_diff' in results:
        filter_diff = results['filter_diff']
        filter_diff[np.isnan(filter_diff)] = -1

        sorted_idx = np.argsort(filter_diff)[::-1]
        filter_diff = filter_diff[sorted_idx]
        keep_idx = [sorted_idx[0]]
        for i in range(0, len(filter_diff) - 1):
            if (filter_diff[i] - filter_diff[i + 1]) < filter_diff_thres * filter_diff[i]:
                keep_idx.append(sorted_idx[i + 1])
            else:
                break
        if show_filters:
            plt.figure()
            sns.set_style('whitegrid')
            plt.plot(np.arange(len(filter_diff)), filter_diff, '--')
            plt.xticks(np.arange(len(filter_diff)), ['filter %d' % i for i in sorted_idx],
                       rotation='vertical')
            plt.ylabel('average cell filter response difference between classes')
            sns.despine()
            plt.savefig(os.path.join(outdir, 'filter_response_differences.pdf'), format='pdf')
            plt.clf()
            plt.close()

    elif 'filter_tau' in results:
        filter_diff = results['filter_tau']
        filter_diff[np.isnan(filter_diff)] = -1

        sorted_idx = np.argsort(filter_diff)[::-1]
        filter_diff = filter_diff[sorted_idx]
        keep_idx = [sorted_idx[0]]
        for i in range(0, len(filter_diff) - 1):
            if (filter_diff[i] - filter_diff[i + 1]) < filter_diff_thres * filter_diff[i]:
                keep_idx.append(sorted_idx[i + 1])
            else:
                break
        if show_filters:
            plt.figure()
            sns.set_style('whitegrid')
            plt.plot(np.arange(len(filter_diff)), filter_diff, '--')
            plt.xticks(np.arange(len(filter_diff)), ['filter %d' % i for i in sorted_idx],
                       rotation='vertical')
            plt.ylabel('Kendalls tau')
            sns.despine()
            plt.savefig(os.path.join(outdir, 'filter_response_differences.pdf'), format='pdf')
            plt.clf()
            plt.close()

    return list(keep_idx)


def plot_filters(results, labels, outdir):
    mkdir_p(outdir)
    nmark = len(labels)
    # plot the filter weights of the best network
    w_best = results['w_best_net']
    idx_except_bias = np.hstack([np.arange(0, nmark), np.arange(nmark + 1, w_best.shape[1])])
    nc = w_best.shape[1] - (nmark + 1)
    labels_except_bias = labels + [f"out {i}" for i in range(nc)]
    w_best = w_best[:, idx_except_bias]
    fig_path = os.path.join(outdir, 'best_net_weights.pdf')
    plot_nn_weights(w_best, labels_except_bias, fig_path, fig_size=(10, 10))
    # plot the filter clustering
    cl = results['clustering_result']
    cl_w = cl['w'][:, idx_except_bias]
    fig_path = os.path.join(outdir, 'clustered_filter_weights.pdf')
    plot_nn_weights(cl_w, labels_except_bias, fig_path, row_linkage=cl['cluster_linkage'],
                    y_labels=cl['cluster_assignments'], fig_size=(10, 10))
    # plot the selected filters
    if results['selected_filters'] is not None:
        w = results['selected_filters'][:, idx_except_bias]
        fig_path = os.path.join(outdir, 'consensus_filter_weights.pdf')
        plot_nn_weights(w, labels_except_bias, fig_path, fig_size=(10, 10))
        filters = results['selected_filters']
    else:
        sys.exit('Consensus filters were not found.')


def plot_nn_weights(w, x_labels, fig_path, row_linkage=None, y_labels=None, fig_size=(10, 3)):
    if y_labels is None:
        y_labels = np.arange(0, w.shape[0])

    if w.shape[0] > 1:
        plt.figure(figsize=fig_size)
        clmap = sns.clustermap(pd.DataFrame(w, columns=x_labels),
                               method='average', metric='cosine', row_linkage=row_linkage,
                               col_cluster=False, robust=True, yticklabels=y_labels, cmap="RdBu_r")
        plt.setp(clmap.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        plt.setp(clmap.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
        clmap.cax.set_visible(True)
    else:
        plt.figure(figsize=(10, 1.5))
        sns.heatmap(pd.DataFrame(w, columns=x_labels), robust=True, yticklabels=y_labels)
        plt.tight_layout()
    plt.savefig(fig_path)
    plt.clf()
    plt.close()

def plot_selected_subset(xc, zc, x, labels, sample_sizes, phenotypes, outdir, suffix,
                         stat_test=None, log_yscale=False,
                         group_a='group A', group_b='group B', group_names=None,
                         regression=False):
    ks_values = []
    nmark = x.shape[1]
    for j in range(nmark):
        ks = stats.ks_2samp(xc[:, j], x[:, j])
        ks_values.append(ks[0])

    # sort markers in decreasing order of KS statistic
    sorted_idx = np.argsort(np.array(ks_values))[::-1]
    sorted_labels = [labels[i] for i in sorted_idx]
    sorted_ks = [('KS = %.2f' % ks_values[i]) for i in sorted_idx]

    fig_path = os.path.join(outdir, 'selected_population_distribution_%s.pdf' % suffix)
    plot_marker_distribution([x[:, sorted_idx], xc[:, sorted_idx]], ['all cells', 'selected'],
                             sorted_labels, grid_size=(9, 9), ks_list=sorted_ks, figsize=(24, 25),
                             colors=['blue', 'red'], fig_path=fig_path, hist=False)

    fig_path = os.path.join(outdir, 'unsorted_selected_population_distribution_%s.pdf' % suffix)

    plot_marker_distribution([x, xc], ['all cells', 'selected'],
                             labels, grid_size=(9, 9), ks_list=np.around(ks_values,decimals=2), figsize=(24, 10),
                             colors=['blue', 'red'], fig_path=fig_path, hist=False)

    # for classification, plot a boxplot of per class frequencies
    # for regression, make a biaxial plot (phenotype vs. frequency)

    if regression:
        frequencies = []
        for i, (n, y_i) in enumerate(zip(sample_sizes, phenotypes)):
            freq = 100. * np.sum(zc == i) / n
            frequencies.append(freq)

        _fig, ax = plt.subplots(figsize=(2.5, 2.5))
        plt.scatter(phenotypes, frequencies)
        if log_yscale:
            ax.set_yscale('log')
        plt.ylim(0, np.max(frequencies) + 1)
        plt.ylabel("selected population frequency (%)")
        plt.xlabel("response variable")
        sns.despine()
        plt.tight_layout()
        fig_path = os.path.join(outdir, 'selected_population_frequencies_%s.pdf' % suffix)
        plt.savefig(fig_path)
        plt.clf()
        plt.close()
    else:
        n_pheno = len(np.unique(phenotypes))
        frequencies = dict()
        for i, (n, y_i) in enumerate(zip(sample_sizes, phenotypes)):
            freq = 100. * np.sum(zc == i) / n
            assert freq <= 100
            print(y_i,freq)
            if y_i in frequencies:
                frequencies[y_i].append(freq)
            else:
                frequencies[y_i] = [freq]
        # optionally, perform a statistical test
        if (n_pheno == 2) and (stat_test is not None):
            freq_a, freq_b = frequencies[0], frequencies[1]
            if stat_test == 'mannwhitneyu':
                _t, pval = stats.mannwhitneyu(freq_a, freq_b)
            elif stat_test == 'ttest':
                _t, pval = stats.ttest_ind(freq_a, freq_b)
            elif stat_test == 'kw':
                _t, pval = stats.kruskal(freq_a, freq_b)                
            elif stat_test == 'ks':
                _t, pval = stats.ks_2samp(freq_a, freq_b)        
            else:
                _t, pval = stats.ttest_ind(freq_a, freq_b)
        else:
            pval = None

        # make a boxplot with error bars
        if group_names is None:
            if n_pheno == 2:
                group_names = [group_a, group_b]
            else:
                group_names = [f"group {y_i + 1}" for y_i in range(n_pheno)]
        box_grade = []
        for y_i, group_name in enumerate(group_names):
            box_grade.extend([group_name] * len(frequencies[y_i]))
        box_data = np.hstack([np.array(frequencies[y_i]) for y_i in range(n_pheno)])
        box = pd.DataFrame(columns=['group', 'selected population frequency (%)'])
        box['group'] = box_grade
        box['selected population frequency (%)'] = box_data

        _fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax = sns.boxplot(x="group", y="selected population frequency (%)", data=box, width=.5,
                         palette=sns.color_palette('Set2'))
        ax = sns.swarmplot(x="group", y="selected population frequency (%)", data=box, color=".25")
        if stat_test is not None:
            ax.text(.45, 1.1, '%s pval = %.2e' % (stat_test, pval), horizontalalignment='center',
                    transform=ax.transAxes, size=8, weight='bold')
        if log_yscale:
            ax.set_yscale('log')
        plt.ylim(np.min(box_data)-1, np.max(box_data) + 1)
        sns.despine()
        plt.tight_layout()
        fig_path = os.path.join(outdir, 'selected_population_frequencies_%s.pdf' % suffix)
        plt.savefig(fig_path)
        plt.clf()
        plt.close()


def plot_marker_distribution(datalist, namelist, labels, grid_size, fig_path=None, letter_size=16,
                             figsize=(9, 9), ks_list=None, colors=None, hist=False):
    nmark = len(labels)
    assert len(datalist) == len(namelist)
    g_i, g_j = grid_size
    sns.set_style('white')
    if colors is None:
        colors = sns.color_palette("Set1", n_colors=len(datalist), desat=.5)

    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(g_i, g_j, wspace=0.1, hspace=.6)
    for i in range(g_i):
        for j in range(g_j):
            seq_index = g_j * i + j
            if seq_index < nmark:
                ax = fig.add_subplot(grid[i, j])
                if ks_list is not None:
                    ax.text(.5, 1.2, labels[seq_index], fontsize=letter_size, ha='center',
                            transform=ax.transAxes)
                    ax.text(.5, 1.02, ks_list[seq_index], fontsize=letter_size - 4, ha='center',
                            transform=ax.transAxes)
                else:
                    ax.text(.5, 1.1, labels[seq_index], fontsize=letter_size, ha='center',
                            transform=ax.transAxes)
                for i_name, (name, x) in enumerate(zip(namelist, datalist)):
                    lower = np.percentile(x[:, seq_index], 0.5)
                    upper = np.percentile(x[:, seq_index], 99.5)
                    if seq_index == nmark - 1:
                        if hist:
                            plt.hist(x[:, seq_index], np.linspace(lower, upper, 10),
                                     color=colors[i_name], label=name, alpha=.5, normed=True)
                        else:
                            sns.kdeplot(x[:, seq_index], shade=True, color=colors[i_name], label=name,
                                        clip=(lower, upper))
                            plt.xlabel('')
                    else:
                        if hist:
                            plt.hist(x[:, seq_index], np.linspace(lower, upper, 10),
                                     color=colors[i_name], label=name, alpha=.5, normed=True)
                        else:
                            sns.kdeplot(x[:, seq_index], shade=True, color=colors[i_name], clip=(lower, upper))
                            plt.xlabel('')
                ax.get_yaxis().set_ticks([])
                # ax.get_xaxis().set_ticks([-2, 0, 2, 4])

    # plt.legend(loc="upper right", prop={'size':letter_size})
    plt.legend(bbox_to_anchor=(1.5, 0.9))
    sns.despine()
    if fig_path is not None:
        plt.savefig(fig_path)
        plt.close()
    else:
        plt.show()

