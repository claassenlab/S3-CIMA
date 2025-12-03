# Generic/Built-in

import numpy as np
import os
import sys
import copy


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, regularizers, optimizers, callbacks


from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# Owned
from utils import combine_samples, normalize_outliers_to_control
from utils import cluster_profiles, keras_param_vector, keras_param_vector_longitudinal
from utils import generate_subsets, generate_biased_subsets, generate_subsets_longitudinal
from utils import get_filters_classification, get_filters_regression
from utils import mkdir_p


class CellCnn:
    """ Creates a CellCnn model.

    Args:
        - ncell :
            Number of cells per multi-cell input.
        - nsubset :
            Total number of multi-cell inputs that will be generated per class, if
            `per_sample` = `False`. Total number of multi-cell inputs that will be generated from
            each input sample, if `per_sample` = `True`.
        - per_sample :
            Whether the `nsubset` argument refers to each class or each input sample.
            For regression problems, it is automatically set to `True`.
        - subset_selection :
            Can be 'random' or 'outlier'. Generate multi-cell inputs uniformly at
            random or biased towards outliers. The latter option is only relevant for detection of
            extremely rare (frequency < 0.1%) cell populations.
        - maxpool_percentages :
            A list specifying candidate percentages of cells that will be max-pooled per
            filter. For instance, mean pooling corresponds to `maxpool_percentages` = `[100]`.
        - nfilter_choice :
            A list specifying candidate numbers of filters for the neural network.
        - scale :
            Whether to z-transform each feature (mean = 0, standard deviation = 1) prior to
            training.
        - quant_normed :
            Whether the input samples have already been pre-processed with quantile
            normalization. In this case, each feature is zero-centered by subtracting 0.5.
        - nrun :
            Number of neural network configurations to try (should be set >= 3).
        - regression :
            Set to `True` for a regression problem. Default is `False`, which corresponds
            to a classification setting.
        - learning_rate :
            Learning rate for the Adam optimization algorithm. If set to `None`,
            learning rates in the range [0.001, 0.01] will be tried out.
        - dropout :
            Whether to use dropout (at each epoch, set a neuron to zero with probability
            `dropout_p`). The default behavior 'auto' uses dropout when `nfilter` > 5.
        - dropout_p :
            The dropout probability.
        - coeff_l1 :
            Coefficient for L1 weight regularization.
        - coeff_l2 :
            Coefficient for L2 weight regularization.
        - max_epochs :
            Maximum number of iterations through the data.
        - patience :
            Number of epochs before early stopping (stops if the validation loss does not
            decrease anymore).
        - dendrogram_cutoff :
            Cutoff for hierarchical clustering of filter weights. Clustering is
            performed using cosine similarity, so the cutof should be in [0, 1]. A lower cutoff will
            generate more clusters.
        - accur_thres :
            Keep filters from models achieving at least this accuracy. If less than 3
            models pass the accuracy threshold, keep filters from the best 3 models.
    """


    def __init__(self, ntime_points=2, ncell=200, nsubset=1000, per_sample=False, subset_selection='random',
                 maxpool_percentages=[0.01, 1, 5, 20, 100], scale=True, quant_normed=False,
                 nfilter_choice=list(range(3, 10)), dropout='auto', dropout_p=.5,
                 coeff_l1=0, coeff_l2=0.0001, learning_rate=None,
                 regression=False, max_epochs=20, patience=5, nrun=15, dendrogram_cutoff=0.4,
                 accur_thres=.95, verbose=1, selection_type='mean', share_filter_weights=False, useRandomHyperparm=False):

        # initialize model attributes
        self.ntime_points=ntime_points
        self.scale = scale
        self.quant_normed = quant_normed
        self.nrun = nrun
        self.regression = regression
        self.ncell = ncell
        self.nsubset = nsubset
        self.per_sample = per_sample
        self.subset_selection = subset_selection
        self.maxpool_percentages = maxpool_percentages
        self.nfilter_choice = nfilter_choice
        self.learning_rate = learning_rate
        self.coeff_l1 = coeff_l1
        self.coeff_l2 = coeff_l2
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.max_epochs = max_epochs
        self.patience = patience
        self.dendrogram_cutoff = dendrogram_cutoff
        self.accur_thres = accur_thres
        self.verbose = verbose
        self.selection_type = selection_type
        self.share_filter_weights = share_filter_weights
        self.useRandomHyperparm = useRandomHyperparm
        self.results = None or dict()



    def fit(self, ntime_points, train_samples, train_phenotypes, outdir, valid_samples=None,
            valid_phenotypes=None, generate_valid_set=True):

        """ Trains a CellCnn model.

        Args:
            - train_samples :
                List with input samples (e.g. cytometry samples) as numpy arrays.
            - train_phenotypes :
                List of phenotypes associated with the samples in `train_samples`.
            - outdir :
                Directory where output will be generated.
            - valid_samples :
                List with samples to be used as validation set while training the network.
            - valid_phenotypes :
                List of phenotypes associated with the samples in `valid_samples`.
            - generate_valid_set :
                If `valid_samples` is not provided, generate a validation set
                from the `train_samples`.

        Returns:
            A trained CellCnn model with the additional attribute `results`. The attribute `results`
            is a dictionary with the following entries:

            - clustering_result : clustered filter weights from all runs achieving \
                validation accuracy above the specified threshold `accur_thres`
            - selected_filters : a consensus filter matrix from the above clustering result
            - best_3_nets : the 3 best models (achieving highest validation accuracy)
            - best_net : the best model
            - w_best_net : filter and output weights of the best model
            - accuracies : list of validation accuracies achieved by different models
            - best_model_index : list index of the best model
            - config : list of neural network configurations used
            - scaler : a z-transform scaler object fitted to the training data
            - n_classes : number of output classes
        """
        res = train_model(ntime_points, train_samples, train_phenotypes, outdir,
                          valid_samples, valid_phenotypes, generate_valid_set,
                          scale=self.scale, nrun=self.nrun, regression=self.regression,
                          ncell=self.ncell, nsubset=self.nsubset, per_sample=self.per_sample,
                          subset_selection=self.subset_selection,
                          maxpool_percentages=self.maxpool_percentages,
                          nfilter_choice=self.nfilter_choice,
                          learning_rate=self.learning_rate,
                          coeff_l1=self.coeff_l1, coeff_l2=self.coeff_l2,
                          dropout=self.dropout, dropout_p=self.dropout_p,
                          max_epochs=self.max_epochs,
                          patience=self.patience, dendrogram_cutoff=self.dendrogram_cutoff,
                          accur_thres=self.accur_thres, verbose=self.verbose,
                          selection_type=self.selection_type,
                          share_filter_weights=self.share_filter_weights,
                          useRandomHyperparm = self.useRandomHyperparm)

        
        self.results = res

        return self

    def predict(self, new_samples, ncell_per_sample=None):

        """ Makes predictions for new samples.

        Args:
            - new_samples :
                List with input samples (numpy arrays) for which predictions will be made.
            - ncell_per_sample :
                Size of the multi-cell inputs (only one multi-cell input is created
                per input sample). If set to None, the size of the multi-cell inputs equals the
                minimum size in `new_samples`.

        Returns:
            y_pred : Phenotype predictions for `new_samples`.
        """

        if ncell_per_sample is None:
            ncell_per_sample = np.min([x.shape[0] for x in new_samples])
        print('Predictions based on multi-cell inputs containing %d cells.' % ncell_per_sample)

        # z-transform the new samples if we did that for the training samples
        scaler = self.results['scaler']
        if isinstance(scaler, dict):
           scaler = scaler[str(self.ntime_points)]

        if scaler is not None:
            new_samples = copy.deepcopy(new_samples)
            new_samples = [scaler.transform(x) for x in new_samples]

        nmark = new_samples[0].shape[1]
        n_classes = self.results['n_classes']

        # get the configuration of the top 3 models
        accuracies = self.results['accuracies']
        sorted_idx = np.argsort(accuracies)[::-1][:3]
        config = self.results['config']

        y_pred = np.zeros((3, len(new_samples), n_classes))
        for i_enum, i in enumerate(sorted_idx):
            nfilter = config['nfilter'][i]
            maxpool_percentage = config['maxpool_percentage'][i]
            ncell_pooled = max(1, int(maxpool_percentage / 100. * ncell_per_sample))

            # build the model architecture
            model = build_model(self.ntime_points,ncell_per_sample, nmark,
                                nfilter=nfilter, coeff_l1=0, coeff_l2=0,
                                k=ncell_pooled, dropout=False, dropout_p=0,
                                regression=self.regression, n_classes=n_classes, lr=0.01, selection_type=self.selection_type,
                                share_filter_weights=self.share_filter_weights)


            # and load the learned filter and output weights
            weights = self.results['best_3_nets'][i_enum]
            
            model.set_weights(weights)

            # select a random subset of `ncell_per_sample` and make predictions
            new_samples = [shuffle(x)[:ncell_per_sample].reshape(1, ncell_per_sample, nmark)
                           for x in new_samples]              
            data_test = np.vstack(new_samples)
            y_pred[i_enum] = model.predict(data_test)
        return np.mean(y_pred, axis=0)

def train_model(ntime_points,train_samples, train_phenotypes, outdir,
                valid_samples=None, valid_phenotypes=None, generate_valid_set=True,
                scale=True, quant_normed=False, nrun=20, regression=False,
                ncell=200, nsubset=1000, per_sample=False, subset_selection='random',
                maxpool_percentages=None, nfilter_choice=None,
                learning_rate=None, coeff_l1=0, coeff_l2=1e-4, dropout='auto', dropout_p=.5,
                max_epochs=20, patience=5,
                dendrogram_cutoff=0.4, accur_thres=.95, verbose=1,
                selection_type = 'mean',
                share_filter_weights=False, useRandomHyperparm=False):

    
    if maxpool_percentages is None:
        maxpool_percentages = [0.01, 1., 5., 20., 100.]
    if nfilter_choice is None:
        nfilter_choice = list(range(3, 10))




    """ Performs a CellCnn analysis """
    mkdir_p(outdir)

    # define samples for each time point
   

    if nrun < 3:
        print('The nrun argument should be >= 3, setting it to 3.')
        nrun = 3

    # copy the list of samples so that they are not modified in-place
    train_samples = copy.deepcopy(train_samples)

    if valid_samples:
        valid_samples = copy.deepcopy(valid_samples)

    # normalize extreme values
    # we assume that 0 corresponds to the control class
    if subset_selection == 'outlier':

       ctrl_list = dict()
       test_list = dict()
       for nt in range(1,ntime_points+1):
           ctrl_list[str(nt)] = [train_samples[str(nt)][i] for i in np.where(np.array(train_phenotypes) == 0)[0]]
           test_list[str(nt)] = [train_samples[str(nt)][i] for i in np.where(np.array(train_phenotypes) != 0)[0]]
           train_samples[str(nt)] = normalize_outliers_to_control(ctrl_list[str(nt)], test_list[str(nt)])
           

       if valid_samples:
           for nt in range(1,ntime_points+1):
               ctrl_list[str(nt)] = [valid_samples[str(nt)][i] for i in np.where(np.array(valid_phenotypes) == 0)[0]]
               test_list[str(nt)] = [valid_samples[str(nt)][i] for i in np.where(np.array(valid_phenotypes) != 0)[0]]
               valid_samples[str(nt)] = normalize_outliers_to_control(ctrl_list[str(nt)], test_list[str(nt)])

    
    X_train, id_train = dict(), dict()
    #X_train_temp = dict()
    X_valid, id_valid = dict(), dict()
    z_scaler=dict()
    X = dict()
    
    train_sample_ids = np.arange(len(train_phenotypes))

    if (not valid_samples) and (not generate_valid_set):
        
        
        for nt in range(1,ntime_points+1):
            X_train[str(nt)], id_train[str(nt)] = combine_samples(train_samples[str(nt)], train_sample_ids)


    elif (not valid_samples) and generate_valid_set:

        valid_phenotypes = train_phenotypes
        

        eval_folds = 5        
        kf = StratifiedKFold(n_splits=eval_folds)
       
        for nt in range(1,ntime_points+1):
            X[str(nt)], sample_id = combine_samples(train_samples[str(nt)], train_sample_ids)
            
            train_indices, valid_indices = next(kf.split(X[str(nt)], sample_id))
            X_train[str(nt)], id_train[str(nt)] = X[str(nt)][train_indices], sample_id[train_indices]
            X_valid[str(nt)], id_valid[str(nt)] = X[str(nt)][valid_indices], sample_id[valid_indices]
     

    else:
        for nt in range(1,ntime_points+1):
            X_train[str(nt)], id_train[str(nt)] = combine_samples(train_samples[str(nt)], train_sample_ids)

        valid_sample_ids = np.arange(len(valid_phenotypes))
        
        for nt in range(1,ntime_points+1):
            X_valid[str(nt)], id_valid[str(nt)] = combine_samples(valid_samples[str(nt)], valid_sample_ids)


    if quant_normed:
        
        for nt in range(1,ntime_points+1):
            z_scaler[str(nt)] = StandardScaler(with_mean=True, with_std=False)
            z_scaler[str(nt)].fit(0.5 * np.ones((1, X_train[str(nt)].shape[1])))
            X_train[str(nt)] = z_scaler[str(nt)].transform(X_train[str(nt)])


    elif scale:

        for nt in range(1,ntime_points+1):
            z_scaler[str(nt)] = StandardScaler(with_mean=True, with_std=False)
            z_scaler[str(nt)].fit(X_train[str(nt)])
            X_train[str(nt)] = z_scaler[str(nt)].transform(X_train[str(nt)])

    else:
        for nt in range(1,ntime_points+1):
            z_scaler[str(nt)] = None     
    
    # an array containing the phenotype for each single cell
   
    y_train = dict()

    for nt in range(1,ntime_points+1):

        X_train[str(nt)], id_train[str(nt)] = shuffle(X_train[str(nt)], id_train[str(nt)])
        train_phenotypes = np.asarray(train_phenotypes)
        y_train[str(nt)] = train_phenotypes[id_train[str(nt)]]        


    if (valid_samples) or generate_valid_set:
        if scale:          
           for nt in range(1,ntime_points+1):    
               X_valid[str(nt)] = z_scaler[str(nt)].transform(X_valid[str(nt)])

        
        
       
        y_valid = dict()

        for nt in range(1,ntime_points+1):

            X_valid[str(nt)], id_valid[str(nt)] = shuffle(X_valid[str(nt)], id_valid[str(nt)])
            valid_phenotypes = np.asarray(valid_phenotypes)
            y_valid[str(nt)] = valid_phenotypes[id_valid[str(nt)]]
    


    # number of measured markers
    nmark = X_train['1'].shape[1]

    # generate multi-cell inputs
    print('Generating multi-cell inputs...')

    
    X_tr, y_tr = dict(), dict()
    X_v, y_v   = dict(), dict()

    if subset_selection == 'outlier':
        # here we assume that class 0 is always the control class
        x_ctrl_train = dict()
        to_keep = dict()        
        x_ctrl_valid = dict()


        for nt in range(1,ntime_points+1):

            x_ctrl_train[str(nt)] = X_train[y_train[str(nt)] == 0]
            to_keep[str(nt)]      = int(0.1 * (X_train[str(nt)].shape[0] / len(train_phenotypes)))            
            nsubset_ctrl = nsubset / np.sum(train_phenotypes == 0)



        # generate a fixed number of subsets per class
        nsubset_biased = [0]
        for pheno in range(1, len(np.unique(train_phenotypes))):
            nsubset_biased.append(nsubset // np.sum(train_phenotypes == pheno))

        
        for nt in range(1,ntime_points+1):

            X_tr[str(nt)], y_tr[str(nt)] = generate_biased_subsets(X_train[str(nt)], train_phenotypes, id_train[str(nt)], x_ctrl_train[str(nt)],
                                                   nsubset_ctrl, nsubset_biased, ncell, to_keep[str(nt)],
                                                   id_ctrl=np.where(train_phenotypes == 0)[0],
                                                   id_biased=np.where(train_phenotypes != 0)[0])

        if (valid_samples) or generate_valid_set:
           
            for nt in range(1,ntime_points+1):
               
               x_ctrl_valid[str(nt)] = X_valid[str(nt)][[y_valid[str(nt)]== 0]]



            nsubset_ctrl = nsubset // np.sum(valid_phenotypes == 0)

            # generate a fixed number of subsets per class
            nsubset_biased = [0]
            for pheno in range(1, len(np.unique(valid_phenotypes))):
                nsubset_biased.append(nsubset // np.sum(valid_phenotypes == pheno))
            

            for nt in range(1,ntime_points+1):
                to_keep[str(nt)] = int(0.1 * (X_valid[str(nt)].shape[0] / len(valid_phenotypes)))
                X_v[str(nt)], y_v[str(nt)] = generate_biased_subsets(X_valid[str(nt)], valid_phenotypes, id_valid[str(nt)], x_ctrl_valid[str(nt)],
                                                                    nsubset_ctrl, nsubset_biased, ncell, to_keep[str(nt)],
                                                                    id_ctrl=np.where(valid_phenotypes == 0)[0],
                                                                    id_biased=np.where(valid_phenotypes != 0)[0])
            
        else:
            
            cut=dict()
            for nt in range(1,ntime_points+1):
               cut[str(nt)] = X_tr[str(nt)].shape[0] // 5
               X_v[str(nt)] = X_tr[str(nt)][:cut[str(nt)]]
               y_v[str(nt)] = y_tr[str(nt)][:cut[str(nt)]]
               X_tr[str(nt)] = X_tr[str(nt)][cut[str(nt)]:]
               y_tr[str(nt)] = y_tr[str(nt)][cut[str(nt)]:]

    else:

        if (ntime_points==1):

               # generate 'nsubset' multi-cell inputs per input sample
            if per_sample:
                X_tr[str(ntime_points)], y_tr = generate_subsets(X_train[str(ntime_points)], train_phenotypes, id_train[str(ntime_points)],
                                              nsubset, ncell, per_sample)
                if (valid_samples is not None) or generate_valid_set:
                    X_v[str(ntime_points)], y_v = generate_subsets(X_valid[str(ntime_points)], valid_phenotypes, id_valid[str(ntime_points)],
                                                nsubset, ncell, per_sample)
            # generate 'nsubset' multi-cell inputs per class
            else:
                nsubset_list = []
                for pheno in range(len(np.unique(train_phenotypes))):
                    nsubset_list.append(nsubset // np.sum(train_phenotypes == pheno))
                   
                X_tr[str(ntime_points)], y_tr = generate_subsets(X_train[str(ntime_points)], train_phenotypes, id_train[str(ntime_points)],
                                              nsubset_list, ncell, per_sample)

               

                if (valid_samples is not None) or generate_valid_set:
                    nsubset_list = []
                    for pheno in range(len(np.unique(valid_phenotypes))):
                        nsubset_list.append(nsubset // np.sum(valid_phenotypes == pheno))
                    X_v[str(ntime_points)], y_v = generate_subsets(X_valid[str(ntime_points)], valid_phenotypes, id_valid[str(ntime_points)],
                                                nsubset_list, ncell, per_sample) 
                    
            
            X_tr[str(ntime_points)] = np.swapaxes(X_tr[str(ntime_points)], 2, 1)
            X_v[str(ntime_points)] = np.swapaxes(X_v[str(ntime_points)], 2, 1)
        
        else:

            X_tr, y_tr = generate_subsets_longitudinal(ntime_points,X_train,id_train, train_phenotypes,nsubset, ncell) 

            if (valid_samples is not None) or generate_valid_set:

                X_v, y_v = generate_subsets_longitudinal(ntime_points, X_valid,id_valid, valid_phenotypes,nsubset, ncell)

    print('Done.')
    ## neural network configuration ##
    # batch size
    bs = 200
    n_classes = 1

    if not regression:
        n_classes = len(np.unique(train_phenotypes))
        y_tr = keras.utils.to_categorical(y_tr, n_classes)
        y_v = keras.utils.to_categorical(y_v, n_classes)
    
    # train some neural networks with different parameter configurations
    accuracies = np.zeros(nrun)
    w_store = dict()
    config = dict()
    config['nfilter'] = []
    config['learning_rate'] = []
    config['maxpool_percentage'] = []
    lr = learning_rate

    #strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())

    for irun in range(nrun):
        if verbose:
            print('training network: %d' % (irun + 1))
        if learning_rate is None:
            lr = 10 ** np.random.uniform(-3,-2) 
            config['learning_rate'].append(lr)
        
        print('Learning rate: %d' % (lr * 1000))
        nfilter = np.random.choice(nfilter_choice)
        config['nfilter'].append(nfilter)
        print('Number of filters: %d' % nfilter)
        
        mp = maxpool_percentages[irun % len(maxpool_percentages)]
        config['maxpool_percentage'].append(mp)
        k = max(1, int(mp / 100. * ncell))
        print('Cells pooled: %d' % k)

        model = build_model(ntime_points, ncell, nmark, nfilter,
                                coeff_l1, coeff_l2, k,
                                dropout, dropout_p, regression, n_classes, lr, selection_type=selection_type,
                                share_filter_weights=share_filter_weights)
             

        filepath = os.path.join(outdir, 'nnet_run_%d.hdf5' % irun)
        try:
            if not regression:
                if (ntime_points==1):
                    train_data_set = X_tr[str(nt)]
                    valid_data_set = X_v[str(nt)]
                else:    
                    train_data_set=[]
                    valid_data_set=[]
                
                    for nt in range(1,ntime_points+1):     

                        train_data_set.append(float32(X_tr[str(nt)]))
                        valid_data_set.append(float32(X_v[str(nt)]))

                
                check = callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True,mode='auto')
                earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='auto')
            
                model.fit(train_data_set, int32(y_tr),batch_size=bs,
                          epochs=max_epochs,callbacks=[check, earlyStopping],
                          validation_data=(valid_data_set, int32(y_v)), verbose=verbose)

            else:
                train_data_set = X_tr[str(nt)]
                valid_data_set = X_v[str(nt)]
                check = callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True,mode='auto')
                earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='auto')
                model.fit(float32(train_data_set), float32(y_tr), batch_size=bs,
                          epochs=max_epochs,callbacks=[check, earlyStopping],
                           validation_data=(float32(valid_data_set), float32(y_v)), verbose=verbose)

            # load the model from the epoch with highest validation accuracy
            model.load_weights(filepath)
            model.save(outdir+'/best_model.h5') 

            if not regression:
                valid_metric = model.evaluate(valid_data_set,int32(y_v))[-1]
                print('Best validation accuracy: %.2f' % valid_metric)
                accuracies[irun] = valid_metric

            else:
                train_metric = model.evaluate(train_data_set,int32(y_tr), batch_size=bs)
                print('Best train loss: %.2f' % train_metric)
                valid_metric = model.evaluate(valid_data_set,int32(y_v), batch_size=bs)
                print('Best validation loss: %.2f' % valid_metric)
                accuracies[irun] = - valid_metric

            # extract the network parameters
            w_store[irun] = model.get_weights()

        except Exception as e:
            sys.stderr.write('An exception was raised during training the network.\n')
            sys.stderr.write(str(e) + '\n')

    #print('figures are generated .. ', os.path.join(outdir, filename))
    
    # the top 3 performing networks
    model_sorted_idx = np.argsort(accuracies)[::-1][:3]
    best_3_nets = [w_store[i] for i in model_sorted_idx]
    best_net = best_3_nets[0]
    best_accuracy_idx = model_sorted_idx[0]
    print('Best model validation accuracy ',np.sort(accuracies)[-1])
    print('std of validation accuracies across models ',np.std(accuracies))
    
    
    # weights from the best-performing network
    if (ntime_points==1):
        w_best_net = keras_param_vector(best_net)
        w_cons, cluster_res = cluster_profiles(w_store, nmark, accuracies, accur_thres,
                                           dendrogram_cutoff=dendrogram_cutoff)
        
        results = { 'clustering_result': cluster_res,
                    'selected_filters': w_cons,
                    'best_net': best_net,
                    'best_3_nets': best_3_nets,
                    'w_best_net': w_best_net,
                    'accuracies': accuracies,
                    'best_model_index': best_accuracy_idx,
                    'config': config,
                    'scaler': z_scaler[str(ntime_points)],
                    'n_classes' : n_classes} 

        if (valid_samples is not None) and (w_cons is not None):
            maxpool_percentage = config['maxpool_percentage'][best_accuracy_idx]
            if regression:
                tau = get_filters_regression(w_cons, z_scaler[str(ntime_points)], valid_samples[str(ntime_points)], valid_phenotypes,
                                         maxpool_percentage)
                results['filter_tau'] = tau

            else:
                filter_diff = get_filters_classification(w_cons, z_scaler[str(ntime_points)], valid_samples[str(ntime_points)],
                                                     valid_phenotypes, maxpool_percentage)
                results['filter_diff'] = filter_diff
    
    else:
        w_best_net=dict()
        share_filter_weights=False
        if share_filter_weights:
            nt=1
            w_best_net[str(nt)] = keras_param_vector_longitudinal(best_net, ntime_points, time_index=None, share_filter_weights=share_filter_weights)
        else:
           for nt in range(1,ntime_points+1): 
            w_best_net[str(nt)]=keras_param_vector_longitudinal(best_net, ntime_points, time_index=nt-1, share_filter_weights=share_filter_weights)
        

    
    
    return results 

def float32(k):
    return np.cast['float32'](k)

def int32(k):
    return np.cast['int32'](k)

def select_top(x, k, type='mean'):
    if type == 'mean':
        # dimension of x (nbatch (axis = 0), cells(row) (axis = 1) , number of filters (axis = 2 or -1))
        # the following operation will sort the rows (ascending) (axis = 1), take the last k rows, for all the batches and for all the filters
        # after that, take the mean of the the last k rows for each filter        
        return tf.reduce_mean(tf.sort(x, axis=1, direction='DESCENDING')[:, :k, :], axis=1)
    elif type == 'max':
        return tf.reduce_max(tf.sort(x, axis=1, direction='DESCENDING')[:, :k, :], axis=1)
    else:
        raise Exception('Selection type not defined.')

def similarity(x, k):
    eps = tf.math.epsilon()
    a = x[:, :k]
    b = x[:, k:]
    return tf.math.log(a + eps) - tf.math.log(b + eps)



def build_model(ntime_points, ncell, nmark, nfilter, coeff_l1, coeff_l2, k, dropout, dropout_p, regression, n_classes, lr=0.01,
                share_filter_weights=False, selection_type='mean'):
    

    """ Builds the neural network architecture """
    x_t=dict()          # dictionary for the inputs
    pool=dict()         # dictionary for pooled response from the filters
    x_t_conv=dict()     # dictionary to store the convolution results applied to the inputs
    #time_residuals_simple = True
    
    

    # the input layer

    for nt in range(1, ntime_points+1):

        x_t[str(nt)] = keras.Input(shape=(ncell, nmark), name='Input'+str(nt))
        
    # the filters

    if share_filter_weights:

        conv_share = layers.Conv1D(filters=nfilter,
                         kernel_size=1,
                         activation='relu',
                         kernel_initializer=initializers.RandomUniform(),
                         kernel_regularizer=regularizers.l1_l2(l1=coeff_l1, l2=coeff_l2),
                         name='conv_share')

        for nt in range(1, ntime_points+1):
            x_t_conv[str(nt)]=conv_share(x_t[str(nt)])

    else:
        #x_t_act=dict()
        for nt in range(1, ntime_points+1):
            conv_not_share = layers.Conv1D(filters=nfilter,
                                           kernel_size=1,
                                           activation='relu',
                                           kernel_initializer=initializers.RandomUniform(),
                                           kernel_regularizer=regularizers.l1_l2(l1=coeff_l1, l2=coeff_l2),
                                           name='conv_not_share'+str(nt))

            
            x_t_conv[str(nt)]=conv_not_share(x_t[str(nt)])

    # the cell grouping part
    pool[str(nt)] = layers.Lambda(select_top, output_shape=(nfilter,), arguments={'k': k, 'type' : selection_type})(x_t_conv[str(nt)])

            # possibly add dropout
    if dropout or ((dropout == 'auto') and (nfilter > 5)):
        pool[str(nt)] = layers.Dropout(dropout_p)(pool[str(nt)])

    
    concat_pool =[]
    
    for nt in range(1, ntime_points+1):

        concat_pool.append(pool[str(nt)])

    if (ntime_points>1):
       merged_pool = concat_pool

    else:
       merged_pool=pool[str(ntime_points)]
       #merged_pool = Flatten()(merged_pool)
    

    if (ntime_points==1):

       if not regression:        
            output = layers.Dense(units=n_classes,
                                  activation='softmax',
                                  kernel_initializer=initializers.RandomUniform(),
                                  kernel_regularizer=regularizers.l1_l2(l1=coeff_l1, l2=coeff_l2),
                                  name='output')(merged_pool)
    

       else:
            output = layers.Dense(units=1,
                                  activation='linear',
                                  kernel_initializer=initializers.RandomUniform(),
                                  kernel_regularizer=regularizers.l1_l2(l1=coeff_l1, l2=coeff_l2),
                                  name='output')(merged_pool)
       
       
       model = keras.Model(inputs=x_t[str(ntime_points)], outputs=output) 
    
    else:
       exit('time point is 1') 
                     
    
    #print(model.summary())
    if not regression:
        model.compile(optimizer=optimizers.Adam(learning_rate=lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    else:
        model.compile(optimizer=optimizers.Adam(learning_rate=lr),
              loss='mean_squared_error')

    
    return model



