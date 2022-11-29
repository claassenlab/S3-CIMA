
""" Copyright 2016-2017 ETH Zurich, Eirini Arvaniti and Manfred Claassen.

This module contains utility functions.

"""

# Generic/Built-in
import os
import errno
from collections import Counter
import numpy as np
import copy
import random


# Other Libs
import pandas as pd
import sklearn.utils as sku
from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy import stats
from scipy.sparse import coo_matrix
import flowio
try:
    import igraph
except ImportError:
    pass


# Owned
from  downsample import random_subsample, kmeans_subsample, outlier_subsample
from  downsample import weighted_subsample



# extra arguments accepted for backwards-compatibility (with the fcm-0.9.1 package)
def loadFCS(filename, *args, **kwargs):
    f = flowio.FlowData(filename)
    events = np.reshape(f.events, (-1, f.channel_count))
    channels = []
    for i in range(1, f.channel_count+1):
        key = str(i)
        if 'PnS' in f.channels[key] and f.channels[key]['PnS'] != u' ':
            channels.append(f.channels[key]['PnS'])
        elif 'PnN' in f.channels[key] and f.channels[key]['PnN'] != u' ':
            channels.append(f.channels[key]['PnN'])
        else:
            channels.append('None')
    return FcmData(events, channels)

class FcmData(object):
    def __init__(self, events, channels):
        self.channels = channels
        self.events = events
        self.shape = events.shape

    def __array__(self):
        return self.events


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_data(indir, info, marker_names, do_arcsinh, cofactor):
    fnames, phenotypes = info[:, 0], info[:, 1]
    sample_list = []
    for fname in fnames:
        full_path = os.path.join(indir, fname)
        fcs = loadFCS(full_path, transform=None, auto_comp=False)
        marker_idx = [fcs.channels.index(name) for name in marker_names]
        x = np.asarray(fcs)[:, marker_idx]
        if do_arcsinh:
            x = ftrans(x, cofactor)
        sample_list.append(x)
    return sample_list, list(phenotypes)

def save_results(results, outdir, labels):
    csv_dir = os.path.join(outdir, 'exported_filter_weights')
    mkdir_p(csv_dir)
    nmark = len(labels)
    nc = results['w_best_net'].shape[1] - (nmark+1)
    labels_ = labels + ['constant'] + ['out %d' % i for i in range(nc)]
    w = pd.DataFrame(results['w_best_net'], columns=labels_)
    w.to_csv(os.path.join(csv_dir, 'filters_best_net.csv'), index=False)
    w = pd.DataFrame(results['selected_filters'], columns=labels_)
    w.to_csv(os.path.join(csv_dir, 'filters_consensus.csv'), index=False)
    w = pd.DataFrame(results['clustering_result']['w'], columns=labels_)
    w.to_csv(os.path.join(csv_dir, 'filters_all.csv'), index=False)

def get_items(l, idx):
    return [l[i] for i in idx]

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def ftrans(x, c):
    return np.arcsinh(1./c * x)

def rectify(X):
    return np.max(np.hstack([X.reshape(-1, 1), np.zeros((X.shape[0], 1))]), axis=1)

def relu(x):
    return x * (x > 0)

def combine_samples(data_list, sample_id):
    accum_x, accum_y = [], []
    for x, y in zip(data_list, sample_id):
        accum_x.append(x)
        accum_y.append(y * np.ones(x.shape[0], dtype=int))
    return np.vstack(accum_x), np.hstack(accum_y)

# def keras_param_vector(params):
#     W_t = np.squeeze(params[0], axis=0)
#     nfilters=W_t.shape[1]
#     markers= W_t.shape[0]
#     W=np.reshape(np.transpose(W_t),(nfilters,markers))
#     #W = np.squeeze(params[0])
#     b = params[1]
#     W_out = params[2]
#     #print(W)
#     #print(len(W), len(b.reshape(-1, 1)), len(W_out))
#     # store the (convolutional weights + biases + output weights) per filter
#     W_tot = np.hstack([W, b.reshape(-1, 1), W_out])
#     return W_tot

def keras_param_vector(params):
    W = np.squeeze(params[0])
    b = params[1]
    W_out = params[2]
    # store the (convolutional weights + biases + output weights) per filter
    W_tot = np.hstack([W.T, b.reshape(-1, 1), W_out])
    return W_tot


def keras_param_vector_longitudinal(params, ntime, time_index, share_filter_weights):
    nfilter = len(params[1])
    if share_filter_weights:
        W = np.squeeze(params[0])
        if len(W.shape) < 2:
            W = W.reshape(1, -1)

        b = params[1]

        W_out_ntime = params[-2][:nfilter]

        # store the (convolutional weights + biases + output weights) per filter
        W_tot = np.hstack([W, b.reshape(-1, 1), W_out_ntime])

    else:
        W = np.squeeze(params[2*time_index])
        if len(W.shape) < ntime:
                W = W.reshape(1, -1)
        b = params[2*time_index+1]

        W_out_ntime = params[-ntime][:nfilter]

        # store the (convolutional weights + biases + output weights) per filter
        W_tot = np.hstack([W, b.reshape(-1, 1), W_out_ntime])

    return W_tot



def representative(data, metric='cosine', stop=None):
    if stop is None:
        i = np.argmax(np.sum(pairwise_kernels(data, metric=metric), axis=1))
    else:
        i = np.argmax(np.sum(pairwise_kernels(data[:, :stop], metric=metric), axis=1))
    return data[i]

def cluster_tightness(data, metric='cosine'):
    centroid = np.mean(data, axis=0).reshape(1, -1)
    return np.mean(pairwise_kernels(data, centroid, metric=metric))

def cluster_profiles(param_dict, nmark, accuracies, accur_thres=.99,
                     dendrogram_cutoff=.5):
    accum = []
    # if not at least 3 models reach the accuracy threshold, select the filters from the 3 best
    if np.sort(accuracies)[-3] < accur_thres:
        accur_thres = np.sort(accuracies)[-3]

    # combine filters from multiple models
    for i, params in param_dict.items():
        if accuracies[i] >= accur_thres:
            W_tot = keras_param_vector(params)
            accum.append(W_tot)
    w_strong = np.vstack(accum)

    # perform hierarchical clustering on cosine distances
    Z = linkage(w_strong[:, :nmark+1], 'average', metric='cosine')
    clusters = fcluster(Z, dendrogram_cutoff, criterion='distance') - 1
    c = Counter(clusters)
    cons = []
    for key, val in c.items():
        if val > 1:
            members = w_strong[clusters == key]
            cons.append(representative(members, stop=nmark+1))
    if cons != []:
        cons_profile = np.vstack(cons)
    else:
        cons_profile = w_strong #None ######SB#####
    cl_res = {'w': w_strong, 'cluster_linkage': Z, 'cluster_assignments': clusters}
    return cons_profile, cl_res

def cluster_profiles_longitudinal_shared_weights(param_dict, nmark, accuracies, accur_thres=.99,
                     dendrogram_cutoff=.5):
    accum = []
    # if not at least 3 models reach the accuracy threshold, select the filters from the 3 best
    if np.sort(accuracies)[-3] < accur_thres:
        accur_thres = np.sort(accuracies)[-3]

    # combine filters from multiple models
    for i, params in param_dict.items():
        if accuracies[i] >= accur_thres:
            W_tot = keras_param_vector_longitudinal(params, 0, True)
            accum.append(W_tot)
    w_strong = np.vstack(accum)

    # perform hierarchical clustering on cosine distances
    Z = linkage(w_strong[:, :nmark+1], 'average', metric='cosine')
    clusters = fcluster(Z, dendrogram_cutoff, criterion='distance') - 1
    c = Counter(clusters)
    cons = []
    for key, val in c.items():
        if val > 1:
            members = w_strong[clusters == key]
            cons.append(representative(members, stop=nmark+1))
    if cons != []:
        cons_profile = np.vstack(cons)
    else:
        cons_profile = w_strong
    cl_res = {'w': w_strong, 'cluster_linkage': Z, 'cluster_assignments': clusters}
    return cons_profile, cl_res


def cluster_profiles_longitudinal(param_dict, ind_t1, ind_t2, nmark, accuracies, accur_thres=.99,
                     dendrogram_cutoff=.5):
    accum_t1 = []
    accum_t2 = []
    # if not at least 3 models reach the accuracy threshold, select the filters from the 3 best
    if np.sort(accuracies)[-3] < accur_thres:
        accur_thres = np.sort(accuracies)[-3]

    # combine filters from multiple models
    for i, params in param_dict.items():
        if accuracies[i] >= accur_thres:
            W_tot_t1 = keras_param_vector_longitudinal(params, time_index=ind_t1)
            W_tot_t2 = keras_param_vector_longitudinal(params, time_index=ind_t2)
            accum_t1.append(W_tot_t1)
            accum_t2.append(W_tot_t2)
    w_strong_t1 = np.vstack(accum_t1)
    w_strong_t2 = np.vstack(accum_t2)

    w_strong = np.concatenate([w_strong_t1[:, :nmark+1], w_strong_t2[:, :nmark+1]], axis=1)

    # perform hierarchical clustering on cosine distances
    Z = linkage(w_strong, 'average', metric='cosine')
    clusters = fcluster(Z, dendrogram_cutoff, criterion='distance') - 1
    c = Counter(clusters)
    cons_t1 = []
    cons_t2 = []
    vals = []
    for key, val in c.items():
        if val > 1:
            members_t1 = w_strong_t1[clusters == key]
            members_t2 = w_strong_t2[clusters == key]
            cons_t1.append(representative(members_t1, stop=nmark+1))
            cons_t2.append(representative(members_t2, stop=nmark+1))
            vals.append(val)
    if (cons_t1 != []) and (cons_t2 != []):
        cons_profile_t1 = np.vstack(cons_t1)
        cons_profile_t2 = np.vstack(cons_t2)
    else:
        # cons_profile = None
        cons_profile_t1 = w_strong_t1
        cons_profile_t2 = w_strong_t2
        vals = c.values()
    cl_res_t1 = {'w': w_strong_t1, 'cluster_linkage': Z, 'cluster_assignments': clusters}
    cl_res_t2 = {'w': w_strong_t2, 'cluster_linkage': Z, 'cluster_assignments': clusters}
    return cons_profile_t1, cons_profile_t2, cl_res_t1, cl_res_t2, vals

def normalize_outliers(X, lq=.5, hq=99.5, stop=None):
    if stop is None:
        stop = X.shape[1]
    for jj in range(stop):
        marker_t = X[:, jj]
        low, high = np.percentile(marker_t, lq), np.percentile(marker_t, hq)
        X[marker_t < low, jj] = low
        X[marker_t > high, jj] = high
    return X

def normalize_outliers_to_control(ctrl_list, list2, lq=.5, hq=99.5, stop=None):
    X = np.vstack(ctrl_list)
    accum = []
    if stop is None:
        stop = X.shape[1]

    for xx in ctrl_list + list2:
        for jj in range(stop):
            marker_ctrl = X[:, jj]
            low, high = np.percentile(marker_ctrl, lq), np.percentile(marker_ctrl, hq)
            marker_t = xx[:, jj]
            xx[marker_t < low, jj] = low
            xx[marker_t > high, jj] = high
        accum.append(xx)
    return accum

## Utilities for generating random subsets ##

def filter_per_class(X, y, ylabel):
    return X[np.where(y == ylabel)]

def per_sample_subsets(X, nsubsets, ncell_per_subset, k_init=False):
    nmark = X.shape[1]
    shape = (nsubsets, nmark, ncell_per_subset)
    Xres = np.zeros(shape)

    if not k_init:
        for i in range(nsubsets):
            #X_i = random_subsample(X, ncell_per_subset) ####SB#####
            indices = list(range(ncell_per_subset))
            X_i =  X[indices, :] ##(X, ncell_per_subset)
            Xres[i] = X_i.T
    else:
        for i in range(nsubsets):
            X_i = random_subsample(X, 2000)
            X_i = kmeans_subsample(X_i, ncell_per_subset, random_state=i)
            Xres[i] = X_i.T
    return Xres

def generate_subsets(X, pheno_map, sample_id, nsubsets, ncell,
                     per_sample, k_init=False):
    S = dict()
    n_out = len(np.unique(sample_id))
    
    for ylabel in range(n_out):
        X_i = filter_per_class(X, sample_id, ylabel)
        if per_sample:
            S[ylabel] = per_sample_subsets(X_i, nsubsets, ncell, k_init)
        else:          
            n = nsubsets[pheno_map[ylabel]]
            S[ylabel] = per_sample_subsets(X_i, n, ncell, k_init)

    # mix them
    data_list, y_list = [], []
    for y_i, x_i in S.items():
        data_list.append(x_i)
        y_list.append(pheno_map[y_i] * np.ones(x_i.shape[0], dtype=int))

    Xt = np.vstack(data_list)
    yt = np.hstack(y_list)
    
    #print('Xt_shape', Xt.shape)
    #print('yt len', len(yt))

    Xt, yt = sku.shuffle(Xt, yt)
    return Xt, yt



def generate_subsets_longitudinal(ntime_points, X, sample_id, pheno_map, nsubsets, ncell):
    
    random.seed()
    data_list = dict()
    sample_id_list = dict()

    # print('shape of training size', X['1'].shape)

    # print('sample id and length of sample id',sample_id, len(sample_id['1']))

    # print(np.asarray(sample_id['1']))

    # print(np.asarray(sample_id['1']) == np.unique(sample_id['1']))

    # print(pheno_map)

    #exit()
    #exit('generate subsets longitudinal')


    for nt in range(0,ntime_points):

        data_list[str(nt+1)] = list()
        sample_id_list[str(nt+1)] = list()

    y_list = list()

    n_classes = len(np.unique(pheno_map))
    
    ind_sample_id_pheno = dict()

    for pheno in np.unique(pheno_map):

        ind_pheno = np.where(np.asarray(pheno_map) == pheno)[0]

        #print(ind_pheno, int(nsubsets / n_classes))


        #for j in range(int(nsubsets / n_classes)):
        for j in range(int(nsubsets)):

            for nt in range(0,ntime_points):

                ind_sample_id_pheno[str(nt+1)]= [] 

            counter = 0   

            while not (all([len(i)>= ncell for i in list(ind_sample_id_pheno.values())]) or counter > 100):

                ind_pheno_temp = dict()
                    
                for nt in range(0,ntime_points):
                        ind_pheno_temp[str(nt+1)] = np.random.choice(ind_pheno, 1)


                for nt in range(0,ntime_points):

                     ind_sample_id_pheno[str(nt+1)] = np.where(np.asarray(sample_id[str(nt+1)]) == np.unique(sample_id[str(nt+1)])[ind_pheno_temp[str(nt+1)]])[0]
                     #print('length of sample id pheno:', len(ind_sample_id_pheno[str(nt+1)]))

                counter += 1


            for nt in range(0,ntime_points):
                sample_id_list[str(nt+1)].append(np.unique(sample_id[str(nt+1)])[ind_pheno_temp[str(nt+1)]])
                data_list[str(nt+1)].append(X[str(nt+1)][np.random.choice(ind_sample_id_pheno[str(nt+1)], ncell, replace=False)])

        
            y_list.append(pheno)
    
    Xt = dict()
    for nt in range(0,ntime_points):
        Xt[str(nt+1)] = np.stack(data_list[str(nt+1)])

    yt = np.hstack(y_list)

    for nt in range(0,ntime_points):
       sample_id_list[str(nt+1)] = np.hstack(sample_id_list[str(nt+1)])
      

    for nt in range(0,ntime_points):
                                                                                   
        Xt[str(nt+1)], yt, sample_id_list['t'+str(nt+1)] = sku.shuffle(Xt[str(nt+1)], yt, sample_id_list[str(nt+1)])

    return Xt, yt



def per_sample_biased_subsets(X, x_ctrl, nsubsets, ncell_final, to_keep, ratio_biased):
    nmark = X.shape[1]
    Xres = np.empty((nsubsets, nmark, ncell_final))
    nc_biased = int(ratio_biased * ncell_final)
    nc_unbiased = ncell_final - nc_biased

    for i in range(nsubsets):
        print(i)
        x_unbiased = random_subsample(X, nc_unbiased)
        if (i % 100) == 0:
            x_outlier, outlierness = outlier_subsample(X, x_ctrl, to_keep)
        x_biased = weighted_subsample(x_outlier, outlierness, nc_biased)
        Xres[i] = np.vstack([x_biased, x_unbiased]).T
    return Xres

def generate_biased_subsets(X, pheno_map, sample_id, x_ctrl, nsubset_ctrl, nsubset_biased,
                            ncell_final, to_keep, id_ctrl, id_biased):
    S = dict()
    for ylabel in id_biased:
        X_i = filter_per_class(X, sample_id, ylabel)
        n = nsubset_biased[pheno_map[ylabel]]
        S[ylabel] = per_sample_biased_subsets(X_i, x_ctrl, n,
                                              ncell_final, to_keep, 0.5)
    for ylabel in id_ctrl:
        X_i = filter_per_class(X, sample_id, ylabel)
        S[ylabel] = per_sample_subsets(X_i, nsubset_ctrl, ncell_final, k_init=False)

    # mix them
    data_list, y_list = [], []
    for y_i, x_i in S.items():
        data_list.append(x_i)
        y_list.append(pheno_map[y_i] * np.ones(x_i.shape[0], dtype=int))
    Xt = np.vstack(data_list)
    yt = np.hstack(y_list)
    Xt, yt = sku.shuffle(Xt, yt)
    return Xt, yt

def write_readme_file(params, dir_output, filename='ReadMe.txt'):
    fout = os.path.join(dir_output, filename)
    fo = open(fout, 'w')
    for k, v in params.items():
        fo.write(str(k) + ' = ' + str(v) + '\n\n')
    fo.close()
def single_filter_output(filter_params, valid_samples, mp):
    y_pred = np.zeros(len(valid_samples))
    nmark = valid_samples[0].shape[1]
    w, b = filter_params[:nmark], filter_params[nmark]
    w_out = filter_params[nmark+1:]

    for i, x in enumerate(valid_samples):
        g = relu(np.sum(w.reshape(1, -1) * x, axis=1) + b)
        ntop = max(1, int(mp/100. * x.shape[0]))
        gpool = np.mean(np.sort(g)[-ntop:])
        y_pred[i] = gpool
    return y_pred, np.argmax(w_out)

def single_filter_output_longitudinal(filter_params, valid_samples, mp, selection_type):
    y_pred = np.zeros(len(valid_samples))
    nmark = valid_samples[0].shape[1]
    w, b = filter_params[:nmark], filter_params[nmark]
    w_out_ntime = filter_params[nmark+1:nmark+3]

    for i, x in enumerate(valid_samples):
        g = relu(np.sum(w.reshape(1, -1) * x, axis=1) + b)
        ntop = max(1, int(mp/100. * x.shape[0]))
        # gpool = np.mean(np.sort(g)[-ntop:])
        if selection_type == 'mean':
            gpool = np.mean(np.sort(g)[-ntop:])
        elif selection_type == 'max':
            gpool = np.max(np.sort(g)[-ntop:])
        y_pred[i] = gpool
    return y_pred, np.argmax(w_out_ntime)

def get_filters_classification(filters, scaler, valid_samples, valid_phenotypes, mp):
    y_true = np.array(valid_phenotypes)
    filter_diff = np.zeros(len(filters))

    if scaler is not None:
        valid_samples = copy.deepcopy(valid_samples)
        valid_samples = [scaler.transform(x) for x in valid_samples]

    for i, filter_params in enumerate(filters):
        y_pred, filter_class = single_filter_output(filter_params, valid_samples, mp)
        filter_diff[i] = np.mean(y_pred[y_true == filter_class]) -\
                         np.mean(y_pred[y_true != filter_class])
    return filter_diff

def get_filters_classification_longitudinal(filters_t1, filters_t2, scaler_t1, scaler_t2, valid_samples_t1, valid_samples_t2, valid_phenotypes, mp, t_out, selection_type,
                                            share_filter_weights):
    nfilter = len(filters_t1)
    y_true = np.array(valid_phenotypes)
    filter_diff = np.zeros(nfilter)

    if scaler_t1 is not None:
        valid_samples_t1 = copy.deepcopy(valid_samples_t1)
        valid_samples_t1 = [scaler_t1.transform(x) for x in valid_samples_t1]

    if scaler_t2 is not None:
        valid_samples_t2 = copy.deepcopy(valid_samples_t2)
        valid_samples_t2 = [scaler_t2.transform(x) for x in valid_samples_t2]


    for i in range(nfilter):
        if share_filter_weights:
            filter_params_t1 = filters_t1[i]
            filter_params_t2 = filters_t1[i]
        else:
            filter_params_t1 = filters_t1[i]
            filter_params_t2 = filters_t2[i]

        y_pred_t1, filter_class_t2_t1 = single_filter_output_longitudinal(filter_params_t1, valid_samples_t1, mp, selection_type=selection_type)
        y_pred_t2, filter_class_t2_t1 = single_filter_output_longitudinal(filter_params_t2, valid_samples_t2, mp, selection_type=selection_type)

        if t_out == 0:
            filter_diff[i] = (np.mean(y_pred_t2[y_true == filter_class_t2_t1]) - np.mean(y_pred_t1[y_true == filter_class_t2_t1])) - \
                             (np.mean(y_pred_t2[y_true != filter_class_t2_t1]) - np.mean(y_pred_t1[y_true != filter_class_t2_t1]))
        else:
            raise Exception('Variable t_out out of range.')
    return filter_diff


def get_filters_regression(filters, scaler, valid_samples, valid_phenotypes, mp):
    y_true = np.array(valid_phenotypes)
    filter_tau = np.zeros(len(filters))

    if scaler is not None:
        valid_samples = copy.deepcopy(valid_samples)
        valid_samples = [scaler.transform(x) for x in valid_samples]

    for i, filter_params in enumerate(filters):
        y_pred, _dummy = single_filter_output(filter_params, valid_samples, mp)
        # compute Kendall's tau for filter i
        w_out = filter_params[-1]
        filter_tau[i] = stats.kendalltau(y_true, w_out * y_pred)[0]
    return filter_tau

def get_selected_cells(filter_w, data, scaler=None, filter_response_thres=0,
                       export_continuous=False):
    nmark = data.shape[1]
    if scaler is not None:
        data = scaler.transform(data)
    w, b = filter_w[:nmark], filter_w[nmark]
    g = np.sum(w.reshape(1, -1) * data, axis=1) + b
    if export_continuous:
        g = relu(g).reshape(-1, 1)
        g_thres = (g > filter_response_thres).reshape(-1, 1)
        return np.hstack([g, g_thres])
    else:
        return (g > filter_response_thres).astype(int)

def create_graph(x1, k, g1=None, add_filter_response=False):

    # compute pairwise distances between all points
    # optionally, add cell filter activity as an extra feature
    if add_filter_response:
        x1 = np.hstack([x1, g1.reshape(-1, 1)])

    d = pairwise_distances(x1, metric='euclidean')
    # create a k-NN graph
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]

    # create a weighted adjacency matrix from the distances (use gaussian kernel)
    # code from https://github.com/mdeff/cnn_graph/blob/master/lib/graph.py
    gauss_sigma = np.mean(d[:, -1])**2
    w = np.exp(- d**2 / gauss_sigma)

    # weight matrix
    M = x1.shape[0]
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = w.reshape(M*k)
    W = coo_matrix((V, (I, J)), shape=(M, M))
    W.setdiag(0)
    adj = W.todense()

    # now reweight graph edges according to cell filter response similarity
    #def min_kernel(v):
    #   xv, yv = np.meshgrid(v, v)
    #   return np.minimum(xv, yv)
    #activity_kernel = pairwise_kernels(g1.reshape(-1, 1), g1.reshape(-1, 1), metric="rbf")
    #activity_kernel = min_kernel(g1)
    #adj = np.multiply(activity_kernel, adj)

    # create a graph from the adjacency matrix
    # first add the adges (binary matrix)
    G = igraph.Graph.Adjacency((adj > 0).tolist())
    # specify that the graph is undirected
    G.to_undirected()
    # now add weights to the edges
    G.es['weight'] = adj[adj.nonzero()]
    # sa summary of the graph
    igraph.summary(G)
    return G
