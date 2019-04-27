#!/usr/bin/env python3

"""
model.py

This contains all the classes

MIT License. Copyright 2018 Jiun Y. Yen (jiunyyen@gmail.com)
"""



# Suppress warnings - warnings have been previously evaluated to be okay
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore",category =RuntimeWarning)

# Imports
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, roc_auc_score, f1_score, recall_score
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import minmax_scale
from datetime import datetime
from time import time
from src.iofunc import open_pkl, gen_id
from multiprocessing import Pool
from itertools import repeat
from copy import deepcopy
import pdb


# Classes
class Model:

    def __init__(self, data=None, exclude_links=None, labels=None, verbose=False, columns=None, aim=None):

        """
        The Model class contains all the functions necessary to train/evaluate/predict from data. To do anything, a
        training file is needed, but not required to instantiate.

        :param data: Optional, string or list of strings of the path(s) of data files
        :param exclude_links: Optional, list of string(s) of link names to exclude to building network connections
        :param labels: Optional, list of string(s) of label names to exclude from training
        :param verbose: Optional, Boolean for whether to verbose
        :param columns: Optional, dictionary that defines the columns in the data, keys: 'nodes', 'links', 'labels', 'layers'
        :param aim: Optional, None, 'recall', or 'precision'. For autotuning the round_cutoff for your specific aim
        """

        # Model ID, assigned after training
        self.id = ''

        # Data sources
        self.p_datas = data if isinstance(data, list) or data is None else [data]
        self.train_idx = 0
        self.columns = columns

        # Actions
        self.verbose = verbose

        # Data augmentations
        self.exclude_links = exclude_links
        self.labels = labels
        self.k_neighbors = 2
        self.min_network_size = 1
        self.train_multilayers = False
        self.maxlidxratio = 0.25
        self.minlinkfreq = 1
        self.masklayer = []

        # Model parameters
        self.kfold_cv = 3
        self.n_repeat = 10
        self.metrics_avg = 'weighted'
        self.maxinflidxratio = 0.01
        self.inflidxratio_history = []
        self.round_cutoff = None
        self.round_cutoff_history = []
        self.aim = aim
        self.n_estimators = 100
        self.n_est_history = []
        self.min_impurity_decrease = 0.00001
        self.min_imp_dec_history = []
        self.min_sample_leaf = 2
        self.min_leaf_history = []
        self.param_tune_scale = 0.25

        # Initialize classifiers
        self.clf_net = self.gen_rfc()
        self.unique_link2labels = None
        self.clf_opt = RandomForestClassifier(
            n_estimators=0,
            max_features=None,
            min_impurity_decrease=0.001,
            warm_start=True,
            n_jobs=os.cpu_count(),
        )
        self.clf_opt_trained = False

        # To store Data objects
        self.datas = []

    def gen_rfc(self):

        self.n_estimators = int(self.n_estimators) if self.n_estimators > 0 else 1
        self.min_impurity_decrease = self.min_impurity_decrease if self.min_impurity_decrease > 0 else 0
        self.min_sample_leaf = int(np.ceil(self.min_sample_leaf)) if self.min_sample_leaf > 0 else 1

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=None,
            min_impurity_decrease=self.min_impurity_decrease,
            min_samples_leaf=self.min_sample_leaf,
            class_weight='balanced_subsample',
            n_jobs=os.cpu_count(),
        )

        return clf

    def add_data(self, data, build=True, mimic=None):

        if data:
            print('\n__  Adding data \_______________________________\n')

            p_data = data if isinstance(data, list) else [data]

            for p in p_data:
                self.p_datas.append(p)
                if mimic is None:
                    self.datas.append(self._load_data(p, build=build))
                else:
                    d = self._load_data(p, build=False).mimic(data=mimic)
                    self.datas.append(d.build_data())
                print('    + ' + p)

        return self

    def load_data(self, data=None):

        """
        Once the Model class object has been given at least the path of the training, evaluating, AND/OR predicting data
        , load the data into memory using functions provided by the Data class.
        Evaluating or predicting data will only load if there is training data. This is because the network of the
        training data is required to construct the features from any other data.

        :return:
        """

        if data:
            self.p_datas = data if isinstance(data, list) else [data]

        print('\n__  Loading data \_______________________________\n  \ File(s): %s \n' % '\n'.join(self.p_datas))

        self.datas = []

        for p in self.p_datas:
            self.datas.append(self._load_data(p_data=p))

        return self

    def _load_data(self, p_data, build=True):

        data = Data(
            p_data=p_data,
            exclude_links=self.exclude_links,
            labels=self.labels,
            verbose=self.verbose,
            columns=self.columns,
            masklayer=self.masklayer,
        )
        data.min_network_size = self.min_network_size
        data.k_neighbors = self.k_neighbors
        data.maxlidxratio = self.maxlidxratio
        data.minlinkfreq = self.minlinkfreq
        if build:
            data.build_data()

        return data

    def import_model(self, model_pkg):

        """
        Import the model package constructed using .export_model()
        The model package is a Python dictionary of all the fields in the Model class and the Data class objects.
        A model package should contain a previously trained Model object.

        :param model_pkg: Python dictionary of the fields and values required to rebuild a trained Model.
        :return:
        """

        keys = self.__dict__.keys()
        for k, v in model_pkg.items():
            if k != 'verbose' and k in keys:
                self.__dict__[k] = v
            elif k.startswith('datas_'):
                c = k.split('_')
                idx = int(c[1])
                while len(self.datas) <= idx:
                    self.datas.append(Data(verbose=self.verbose))
                self.datas[int(c[1])].__dict__['_'.join(c[2:])] = v

        return self

    def export_model(self):

        """
        Export the Model class object into a model package, so it can be used to evaluate or predict later.
        The model package is a Python dictionary of all the fields in the Model class and the Data class objects.
        A model package should contain a trained Model object.

        :return: model package, a Python dictionary of the fields and values required to rebuild a trained Model
        """

        model_pkg = dict()

        for k, v in self.__dict__.items():
            if k not in ['datas'] and not k.startswith('_'):
                model_pkg[k] = v

        for i in range(len(self.datas)):
            for k, v in self.datas[i].__dict__.items():
                model_pkg['datas_%d_%s' % (i, k)] = v

        return model_pkg

    def load_model(self, p_mod):

        print('\n__  Loading model \_______________________________\n  \ File: %s \n' % p_mod)
        m_pkg, _ = open_pkl(p_mod)
        self.import_model(m_pkg)

        return self

    def train_cv_layers(self):

        if self.train_multilayers and self.datas[self.train_idx].layer2nidx:

            t0 = time()

            self.labels = self.datas[self.train_idx].labels.copy()

            res_final = pd.DataFrame()
            layers = list(self.datas[self.train_idx].layer2nidx.keys())

            p_train0 = self.p_datas[self.train_idx]
            tid = ''.join(np.random.choice(list('abcdefgh12345678'), 6))
            self.p_datas[self.train_idx] = p_train0.replace('.tsv', '-sptrain-%s.tsv' % tid)
            self.p_datas.append(p_train0.replace('.tsv', '-sptest-%s.tsv' % tid))
            test_idx = len(self.p_datas) - 1

            for layer in layers:

                print('\n------{ test layer: %s }---------------------------------------------\n' % layer)

                # Generate temporary training/testing files
                df = pd.read_table(p_train0)
                df[df[self.columns['layers']] == layer].to_csv(self.p_datas[test_idx], sep='\t')
                df[df[self.columns['layers']] != layer].to_csv(self.p_datas[self.train_idx], sep='\t')
                self.load_data()

                # Train
                self.train_multilayers = True
                res = self.train()
                res[self.columns['layers']] = 'no-' + layer
                res_final = res_final.append(res, ignore_index=True)

                # Eval
                self.train_multilayers = False
                res = self.eval(data=self.datas[test_idx], eval_idx=test_idx)
                res[self.columns['layers']] = layer
                res_final = res_final.append(res, ignore_index=True)

            # Remove temporary train/test files
            os.remove(self.p_datas[self.train_idx])
            os.remove(self.p_datas[test_idx])

            # Unload all loaded datasets and reload original data
            self.load_data(data=p_train0)
            self.train_idx = 0

            # Final training with all data
            self.train_multilayers = True
            res = self.train()
            res[self.columns['layers']] = 'all'
            res_final = res_final.append(res, ignore_index=True)

            print('\n [ CV layers ] Total training time: %.1f minutes\n' % ((time()-t0)/60))

            return res_final

        else:
            if not self.train_multilayers:
                print('  { Cannot perform CV-layers because .train_multilayer=False }')
            if not self.datas[self.train_idx].layer2nidx:
                self.train_multilayers = False
                print('  { Cannot perform CV-layers because .data.layer2nidx is empty }')

        return None

    def train(self):

        """
        This is where the model is trained. tonknows contains 3 base classifiers and a final ensemble model that is
        trained on the results of the 3 classifiers. The 3 classifiers are the neighbor-inferred classifier (INF), the
        link-match classifier (MATCH), and network-trained classifier (NET). In order to train all the classifiers,
        the training data must be partitioned into Train, Eval, and Optimize sets. Multiple iterations of partitioning
        is needed to generate enough data to train the final model. First, the Train and Eval sets are used to train and
        evaluate the 3 base classifiers. Then the Eval and Optmize sets are used to train and evaluate the final model.
        For this train/eval, the Eval set serves as the training set and the Optimize set serves as the evaluation set.
        This is because the Eval set will not only have the original truth, but also the predictions from the 3 base
        classifiers, and these predictions are concatenated to form the features used to train the final model to fit
        to the truth. Thus, the Optimize set, which is data that neither the 3 base classifiers nor the final model has
        seen, will be used to evaluate the overall performance. This final performance evaluation is kept in the
        opt_results Pandas data frame, and would also be saved along with the trained model if the --save option is
        used. Once optimal parameters of the final model is found, the MATCH and NET classifiers will need to be trained
        again with ALL the data this time around. INF classifier does not need to be train because it is essentially a
        simple calculation that can only be done on the spot for each sample.

        There are 2 steps:
        1. Iterative training of final model with cross-validation results of the 3 predictors
        2. Final training of MATCH and NET classifiers with all the data

        :return: opt_results, A Pandas data frame with all the overall performance results from each iteration
        """

        # Step 1 - Obtain optimized weights for final model ------------------------------------------------------------

        t0 = time()

        # Check the training data for potential hazardous problems
        self.check_training_samples()

        opt_results = pd.DataFrame()
        kf_opt = StratifiedKFold(n_splits=self.kfold_cv, shuffle=True)
        rep_str, opt_str = '', ''

        if self.verbose:
            print('\n\n__  TRAINING STEP 1/2 \_______________________________')
            print('  \ Train with reverse %d-fold CV - %d time(s) /\n' % (self.kfold_cv, self.n_repeat))

        for i_rep in range(self.n_repeat):

            if self.verbose:
                rep_str = '\n_/--- Rep %d/%d' % (i_rep + 1, self.n_repeat)

            # Sample clf-net parameters to test
            param = [
                np.random.normal(loc=self.n_estimators,
                                 scale=self.n_estimators*self.param_tune_scale,
                                 size=self.kfold_cv),
                np.random.normal(loc=self.min_impurity_decrease,
                                 scale=self.min_impurity_decrease*self.param_tune_scale,
                                 size=self.kfold_cv),
                np.random.normal(loc=self.min_sample_leaf,
                                 scale=np.ceil(self.min_sample_leaf*self.param_tune_scale),
                                 size=self.kfold_cv),
            ]
            scores = list()

            for j_fold, (opt_idxs, cv_train_idxs) in enumerate(kf_opt.split(X=self.datas[self.train_idx].nidx_train, y=self.datas[self.train_idx].gen_labels(condense_labels=True))):

                if self.verbose:
                    print(rep_str + ' - CV %d/%d ---\_____\n' % (j_fold + 1, self.kfold_cv))

                # set clf-net parameters
                self.n_estimators = param[0][j_fold]
                self.min_impurity_decrease = param[1][j_fold]
                self.min_sample_leaf = param[2][j_fold]
                self.clf_net = self.gen_rfc()

                # Split data
                opt_nidxs = np.array([self.datas[self.train_idx].nidx_train[i] for i in opt_idxs])
                cv_train_nidxs = np.array([self.datas[self.train_idx].nidx_train[i] for i in cv_train_idxs])

                # Partition train/eval nidx for reverse k-fold CV training
                _, _, opt_eval_nidxs, opt_train_nidxs = train_test_split(
                    np.zeros(len(opt_nidxs)),
                    opt_nidxs,
                    test_size= 1/(self.kfold_cv - 1),
                    shuffle=True,
                    stratify=self.datas[self.train_idx].gen_labels(nidxs=opt_nidxs, condense_labels=True))

                # Train clfs
                if self.verbose:
                    print('\n> Training base classifiers ...')
                self._train_clfs(train_nidxs=cv_train_nidxs)

                # Evaluate train with cv_train data
                if self.verbose:
                    print('\n> Evaluating base classifiers with cv_train partition ...')
                self.clfs_predict(nidxs_target=cv_train_nidxs, data=self.datas[self.train_idx], to_eval=True, eval_idx=self.train_idx)

                # Evaluate pre-optimization with opt_train data
                if self.verbose:
                    print('\n> Evaluating base classifiers with cv_eval partition ...')
                cv_res = self.clfs_predict(nidxs_target=opt_train_nidxs, data=self.datas[self.train_idx], to_eval=True, nidxs_train=cv_train_nidxs, eval_idx=self.train_idx)

                # Train clf-opt with opt_train partition results
                if self.verbose:
                    print('\n> Training clf-opt ...')
                self._train_clf_opt(predictions=cv_res)

                # Evaluate clf-opt with opt_eval partition
                if self.verbose:
                    print('\n> Evaluating optimized classifier with opt_test partition ...')
                opt_res = self.clfs_predict(nidxs_target=opt_eval_nidxs, data=self.datas[self.train_idx], to_eval=True, nidxs_train=cv_train_nidxs, eval_idx=self.train_idx)
                opt_results = opt_results.append(opt_res, ignore_index=True)

                # Append score to optimize clf-net parameter
                r = self.scores(opt_res['ytruth'], opt_res['ynet'])
                if not self.aim:
                    scores.append(r['aucroc'])
                else:
                    aim = self.aim.replace('hard', '')
                    scores.append(r[aim])

                # reset link2featidx
                self.datas[self.train_idx].link2featidx = {}

            # Aggregate results from clf-net parameter search
            self._set_clf_net_param(param, scores)

        # STEP 2 - Train final model -----------------------------------------------------------------------------------
        # .clf_opt is already trained through previous iterations by using warm_start

        if self.verbose:
            print('\n__  TRAINING STEP 2/2 \_______________________________')
            print('  \ Train final model with all train data /\n')

        # Train clfs with all the data
        self._train_clfs()

        # Evaluate final clf-opt with all data
        print('\n> Evaluating final classifier ...')
        self.clfs_predict(nidxs_target=self.datas[self.train_idx].nidx_train, to_eval=True, eval_idx=self.train_idx)
        print('** Because this is evaluating with the training data, classifier performances should be very high.')

        # Assign model ID - this is here so that if retrained, it would be known that it is not the same model anymore
        self.id = 'm_%s' % gen_id()

        if self.verbose:
            te = (time() - t0) / 60
            print('\n  Training took %.1f minutes on %d processors' % (te, os.cpu_count()))
            print('\n__                      __________')
            print('  \ Training complete! /\n')

        return opt_results

    def _set_clf_net_param(self, param, scores):

        imax = np.argmax(scores)
        self.n_est_history.append(param[0][imax])
        self.min_imp_dec_history.append(param[1][imax])
        self.min_leaf_history.append(param[2][imax])

        self.n_estimators = np.median(self.n_est_history)
        self.min_impurity_decrease = np.median(self.min_imp_dec_history)
        self.min_sample_leaf = np.median(self.min_leaf_history)

        self.clf_net = self.gen_rfc()

        if self.verbose:
            print('\n__ clf-net param auto-tune \_______')
            print(' clf-net.n_estimators = %d' % self.n_estimators)
            print(' clf-net.min_impurity_decrease = %.7f' % self.min_impurity_decrease)
            print(' clf-net.min_sample_leaf = %d' % self.min_sample_leaf)

        return

    def check_training_samples(self):

        """
        Checking to see if there are extremely under-represented labels (less than the number of K-fold partition),
        because, with the current implementation, this can lead to dimension mismatch error because "warm-start" is used
        in training the final random forest classifier.

        :return:
        """

        yidx = np.sum(self.datas[self.train_idx].gen_labels(), axis=0) < self.kfold_cv
        if np.any(yidx):
            xlist = ','.join(np.array(self.datas[self.train_idx].labels)[yidx])
            print('\n  *** WARNING ***\n  There are labels with very few samples: %s' % xlist)
            print('  If encounter chaotic errors, consider excluding these labels using --excludeloc %s\n' % xlist)

        return

    def eval(self, data=None, eval_idx=None):

        """
        Evaluate any samples in a Data object that has label information.

        :param data: Optional. A Data class object. Must be loaded with data by running Data.build(). If None, then
        the first non-train data will be used. If nothnig is in that, then no evaluation.
        :param eval_idx: index of loaded evaluating dataset
        :return: A dictionary containing predictions from all the base classifiers and the final model
        """

        if data is None:
            if len(self.datas) > 1:
                for eval_idx in range(len(self.datas)):
                    if eval_idx != self.train_idx:
                        data = self.datas[eval_idx]
                        break
                print('\n { Evaluating with %s }\n' % data.p_data)

            else:
                data = self.datas[self.train_idx]
                print('\n { Evaluating with training data }\n')

        if eval_idx is None:
            if len(self.datas) > 1:
                for eval_idx in range(len(self.datas)):
                    if eval_idx != self.train_idx and data.p_data == self.datas[eval_idx].p_data:
                        break
            else:
                eval_idx = self.train_idx

        predictions = None
        if data.p_data and os.path.isfile(data.p_data):
            print('\n_/ EVALUATION \_______________________________')
            print(' \ Data: %s' % data.p_data)

            if not data.nidx_train:
                print('\n { Cannot evaluate with this data - no node with label }\n')
                return None

            # Check if data is multilayer
            if self.train_multilayers and not data.layer2nidx:
                self.train_multilayers = False
                if self.verbose:
                    print('\n { Data is not multilayered, setting .train_multilayers=False }')

            n_total = len(data.nidx_train)
            print('  Found %d nodes with labels' % n_total)

            data.composition()

            predictions = self.clfs_predict(nidxs_target=data.nidx_train, data=data, to_eval=True, eval_idx=eval_idx)

        else:
            print('\n { No evaluation dataset }\n')

        return predictions

    def predict_from_param(self, param, write=True):

        results = []
        for p in param['p_datas']:
            p = param['datamod'](p) if 'datamod' in param else p
            self.load_model(p_mod=param['model'])
            self.add_data(data=p, mimic=self.datas[self.train_idx])
            res = self.predict(data=self.datas[-1], write=write)
            res['model'] = param['model']
            results.append(res)

        return results

    def predict(self, data=None, write=True, p_out=None, d_out=None, pred_idx=None):

        """
        The is predicting with a prediction data or the specified Data object. If there are nodes with label
        labels, they will be used to perform an evaluation to estimate how the model could perform for this dataset.
        A report will also be generated with the final predictions for all the nodes, including the labeled and the
        non-labeled.

        :param data: Optional, a Data class object that contains samples to be predicted for. If None, first non-train
        data is used
        :param p_out: Optional, String that specifies the file path of the report to be generated
        :param write: Optional, whether to write predictions to file
        :param pred_idx: Optional, index of the loaded predicting dataset
        :return: Two variables: a dictionary containing predictions from all the base classifiers and the final model,
        and the String of the report file path
        """

        if data is None:
            if len(self.datas) > 1:
                for pred_idx in range(len(self.datas)):
                    if pred_idx != self.train_idx:
                        data = self.datas[pred_idx]
                        break
                print('\n { Evaluating with %s }\n' % data.p_data)

            else:
                data = self.datas[self.train_idx]
                print('\n { Evaluating with training data }\n')

        if pred_idx is None:
            if len(self.datas) > 1:
                for pred_idx in range(len(self.datas)):
                    if pred_idx != self.train_idx and data.p_data == self.datas[pred_idx].p_data:
                        break
            else:
                pred_idx = self.train_idx

        result = {'p_data': data.p_data, 'p_out':'', 'pred':None, 'eval':None}
        if data.p_data and os.path.isfile(data.p_data):
            print('\n_/ PREDICTION \_______________________________')
            print(' \ Data: %s' % data.p_data)

            if not data.nidx_pred:
                print('\n { Nothing to predict for - no node without labels }\n')
                return p_out

            evaluations = None
            if data.nidx_train:
                print('\n---{ Estimating performance with labeled data }---')
                evaluations = self.eval(data=data)
                result['eval'] = evaluations

            n_total = len(data.nidx_pred)
            print('\n---{ Predicting for %d nodes }---\n' % n_total)

            # Check if data is multilayer
            if self.train_multilayers and not data.layer2nidx:
                self.train_multilayers = False
                if self.verbose:
                    print('\n { Data is not multilayered, setting .train_multilayers=False }')

            # Predict
            predictions = self.clfs_predict(nidxs_target=data.nidx_pred, data=data, eval_idx=pred_idx)
            result['pred'] = predictions

            # Write results
            if write:
                p_out = self.write_predictions(predictions=predictions, evaluations=evaluations, data=data, p_out=p_out, d_out=d_out)
                result['p_out'] = p_out

        else:
            print('\n { No evaluation dataset }\n')

        return result

    def write_predictions(self, predictions, evaluations=None, data=None, p_out=None, d_out=None):

        """
        Predictions are written into a .tsv file (tab-delimited). First column lists the nodes' IDs. Column 2 has the
        ground truths at the bottom. Column 3 has the final predictions. Column 4 has predictions and ground truths when
        available (i.e. for the samples with ground truths, those truths are used). Column 5 shows merged predictions
        and ground truths.

        :param predictions:
        :param evaluations:
        :param data:
        :param p_out:
        :param d_out:
        :return: String, output file path
        """

        if not d_out:
            d_out = os.path.split(data.p_data)[0]
        elif not os.path.isdir(d_out):
            os.makedirs(d_out)
            if self.verbose:
                print('\nCreated folder: %s ' % d_out)

        d_out += '/' if not d_out.endswith('/') else ''

        timestamp = datetime.now()
        if not p_out:
            tmp = data.p_data.replace('.tsv', '-tonknows_pred-%s-%s.tsv' % (timestamp.strftime('%Y%m%d%H%M%S'), ''.join(np.random.choice(list('abcdef123456'), 6))))
            p_out = d_out + os.path.split(tmp)[1]

        labels = np.array(data.labels)

        # compile predictions
        yopts = ['' for _ in range(len(data.nodes))]

        # compile baseline
        ybkgs = ['' for _ in range(len(data.nodes))]

        for idx, yopt, ybkg in zip(predictions['nidxs'], predictions['yopt'], predictions['ybkg']):
            yopts[idx] = '/'.join(labels[np.array(self._round(yopt), dtype=bool)])
            ybkgs[idx] = '/'.join(labels[np.array(self._round(ybkg), dtype=bool)])

        # compile known
        if evaluations:
            for idx, yopt, ytruth, ybkg in zip(evaluations['nidxs'], evaluations['yopt'], evaluations['ytruth'], predictions['ybkg']):
                yopt_bool = np.array(self._round(yopt), dtype=bool)
                truth_bool = np.array(ytruth, dtype=bool)
                yopts[idx] = '/'.join(labels[yopt_bool | truth_bool])
                ybkgs[idx] = '/'.join(labels[np.array(self._round(ybkg), dtype=bool)])

        # Add truth and predictions to dataframe
        df = data.df.assign(predictions=yopts)

        # Add baseline to dataframe
        df = df.assign(baseline=ybkgs)

        # Write predictions
        df.to_csv(p_out, sep='\t', index=False)

        print('Results written to %s\n' % p_out)

        return p_out

    def clfs_predict(self, nidxs_target, data=None, to_eval=False, fill=True, nidxs_train=None, eval_idx=None):

        """
        This runs predictions with all base classifiers and the final model for the nodes specified through their
        indices in a Data object. It is possible that the final model does not predict any labels for a sample
        either because the base classifiers cannot work with such sample or the predictions from the base classifiers
        does not lead to any labels predicted by the final model. In the latter case, if any of the base
        classifiers do generate a prediction, this function could attempt to "fill" this type of samples with the joint
        predictions from all the base classifiers. The proportion of data that has labels predicted for (coverage)
        and "filled" will be displayed if verbose is enabled.

        :param nidxs_target: A Python list of indices of nodes in the Data object
        :param data: Optional, Data object loaded with dataset to predict for. If None, then .train_data is used
        :param to_eval: Optional, Boolean to specify whether this is running an evaluation or prediction
        :param fill: Optional, Boolean to specify whether to use predictions from the base classifiers to fill samples
        that the final model did not predict any labels for. Even with True, it does not guarantee a prediction
        will be given, since the base classifiers may not be able to predict for the sample or may not predict any
        labels for the sample as well.
        :param nidxs_train: nodes used in training
        :param eval_idx: index of data to evaluate
        :return: A dictionary containing predictions from all the base classifiers and the final model
        """

        if data is None:
            data = self.datas[self.train_idx]
        train_idx0 = self.train_idx

        if eval_idx is None:
            for i, d in enumerate(self.datas):
                if i != self.train_idx and data.p_data == d.p_data:
                    eval_idx = i
                    break

            # Still doesn't find the right data
            if eval_idx is None:
                print('\n  { Cannot find eval data }\n')
                return None

        # Combine training and testing data into layers of new Data object used to retrain clf-net during expansion
        # The training and test data will be in different layers
        # This is so training data is not lost during expansive training iterations
        self.datas.append(self.blend_data(idxs=list({train_idx0, eval_idx})))
        blend_idx = len(self.datas) - 1
        if nidxs_train is not None:
            nidxs_train_blend = [self.datas[blend_idx].nidxconvert[train_idx0][n] for n in nidxs_train]
        else:
            nidxs_train_blend = [self.datas[blend_idx].nidxconvert[train_idx0][n] for n in self.datas[train_idx0].nidx_train]

        # Compute expansion path
        path = data.recruit(nidxs_remain=set(nidxs_target))
        n_steps = len(path)
        print('\n  { Network expansion path contains %d steps }' % n_steps)

        # Retain original base-classifiers
        clf_pkg = self.archive_clfs()

        # Retain original multilayer state and change to train multilayers
        train_multilayers0 = self.train_multilayers
        self.train_multilayers = True

        nidxs = []
        predictions = None
        for istep in range(n_steps):

            nidxs += [self.datas[blend_idx].nidxconvert[eval_idx][n] for n in path[istep]['nidxs']]
            print('\n[ Step %d/%d ]  %d nodes / %d expandable links' % (istep+1, n_steps, len(path[istep]['nidxs']), len(path[istep]['links'])))

            # Predict
            predictions = self.clf_all_predict(nidx_target=nidxs, data=self.datas[blend_idx])

            # Baseline
            y_bkg_pred = self.bkg_predict(n_samples=len(nidxs), data=self.datas[blend_idx])
            predictions['ybkg'] = y_bkg_pred

            # Predict with clf-opt if trained
            if self.clf_opt_trained:
                X = self._construct_clf_opt_X(predictions)
                predictions['yopt'] = self._predict_proba(self.clf_opt, X)

                # Fill (fill==True) what clf-opt did not predict with the joint solutions from base classifiers
                if fill:
                    coverage0 = self.calc_coverage(predictions['yopt'])
                    filling_coverage = 0.
                    if coverage0 < 1.:
                        i_missing = np.invert(np.any(self._round(predictions['yopt']), axis=1))
                        y_fill = self._y_merge(predictions=predictions, i=i_missing)
                        filling_coverage = self.calc_coverage(y_fill) * len(y_fill) / len(i_missing)
                        predictions['yopt'][i_missing] = y_fill
                        predictions['fill'] = filling_coverage

                    if self.verbose:
                        coverage1 = self.calc_coverage(predictions['yopt'])
                        print('\n[ clf-opt ]\n  no-fill coverage: {:.1%}\n  filling: {:.1%}\n  filled coverage: {:.1%}\n'.format(coverage0, filling_coverage, coverage1))

            # Show scores
            self._print_eval(predictions=predictions, to_eval=to_eval)

            if (istep + 1) < n_steps and path[istep]['links']:

                new_links = list(path[istep]['links'])

                # Set training data index to the blended data
                self.train_idx = blend_idx

                # find expand features with evaluating data then set features in the blended data
                if self.verbose:
                    print('\n[ Expanding ] Evaluating %d links' % len(new_links))
                r = self.datas[blend_idx].eval_lidxs(lidxs=[self.datas[blend_idx].link2lidx[l] for l in new_links], nidxs=nidxs_train_blend)
                accepted_links = [new_links[i] for i, b in enumerate(r) if b]
                if self.verbose:
                    print('[ Expanding ] Accepting %d links' % len(accepted_links))
                n = len(self.datas[blend_idx].link2featidx)
                self.datas[blend_idx].link2featidx.update({link: n + i for i, link in enumerate(accepted_links)})
                if accepted_links and self.verbose:
                    print('[ Expanding ] Expanded features %d -> %d' % (n, len(self.datas[blend_idx].link2featidx)))

                # update labels with predictions in the blended data
                if self.verbose:
                    print('[ Expanding ] Updating labels')
                if 'yopt' in predictions:
                    yp = self._round(predictions['yopt'])
                else:
                    yp = self._y_merge(predictions=predictions)
                nidxs_conv = nidxs.copy()
                labels0 = self.datas[blend_idx].node_labels.copy()
                self.datas[blend_idx].update_labels(nidxs=nidxs_conv, y=yp)

                # Compile all training nodes, which include nodes previously trained and now training in expansion
                if nidxs_train is not None:
                    nidxs_conv += nidxs_train_blend

                # retrain model with predictions and features from this expansion
                if self.verbose:
                    print('[ Expanding ] Training on expanded network')
                self._train_clfs(train_nidxs=nidxs_conv)

                # reset training data index and labels
                self.train_idx = train_idx0
                self.datas[blend_idx].node_labels = labels0.copy()

        # Remove blended data
        self.datas.pop(blend_idx)

        # Restore base-classifiers
        self.restore_clfs(clf_pkg)

        # Restore original multilayer state
        self.train_multilayers = train_multilayers0

        # Node indices
        predictions['nidxs'] = np.array([i for j in path for i in j['nidxs']])

        return predictions

    def _y_merge(self, predictions, i=None):

        if i is None:
            i = np.ones(len(predictions['yinf']), dtype=bool)

        inf_fill = predictions['yinf'][i]
        match_fill = predictions['ymatch'][i]
        net_fill = predictions['ynet'][i]
        y_merge = (self._round(inf_fill) > 0) | (self._round(match_fill) > 0) | (self._round(net_fill) > 0)

        return y_merge

    def _print_eval(self, predictions, to_eval=False):

        if self.verbose:
            if to_eval:
                # header
                print('__/ Evaluation \_______________________________________')
                print('|  Clf   AUC-ROC    F1    Precision  Recall  Coverage |')
                print('| -----  -------  ------  ---------  ------  -------- |')

                # baseline
                r = self.scores(y=predictions['ytruth'], y_pred=predictions['ybkg'])
                coverage = self.calc_coverage(predictions['ybkg'])
                print('| {:<5}  {:<7.4f}  {:<6.4f}  {:<9.4f}  {:<6.4f}  {:>8.1%} |'.format(
                    'BKG', r['aucroc'], r['f1'], r['precision'], r['recall'], coverage))

                # clf-inf
                idx = predictions['inf']
                r = self.scores(y=predictions['ytruth'][idx], y_pred=predictions['yinf'][idx])
                coverage = self.calc_coverage(predictions['yinf'])
                print('| {:<5}  {:<7.4f}  {:<6.4f}  {:<9.4f}  {:<6.4f}  {:>8.1%} |'.format(
                    'INF', r['aucroc'], r['f1'], r['precision'], r['recall'], coverage))

                # clf-match
                idx = predictions['match']
                r = self.scores(y=predictions['ytruth'][idx], y_pred=predictions['ymatch'][idx])
                coverage = self.calc_coverage(predictions['ymatch'])
                print('| {:<5}  {:<7.4f}  {:<6.4f}  {:<9.4f}  {:<6.4f}  {:>8.1%} |'.format(
                    'MATCH', r['aucroc'], r['f1'], r['precision'], r['recall'], coverage))

                # clf-net
                idx = predictions['net']
                r = self.scores(y=predictions['ytruth'][idx], y_pred=predictions['ynet'][idx])
                coverage = self.calc_coverage(predictions['ynet'])
                print('| {:<5}  {:<7.4f}  {:<6.4f}  {:<9.4f}  {:<6.4f}  {:>8.1%} |'.format(
                    'NET', r['aucroc'], r['f1'], r['precision'], r['recall'], coverage))

                # clf-opt
                if 'yopt' in predictions:
                    r = self.scores(y=predictions['ytruth'], y_pred=predictions['yopt'])
                    coverage = self.calc_coverage(predictions['yopt'])
                    print('| {:<5}  {:<7.4f}  {:<6.4f}  {:<9.4f}  {:<6.4f}  {:>8.1%} |'.format(
                        '*OPT', r['aucroc'], r['f1'], r['precision'], r['recall'], coverage))

                if not np.all(r['label_ratios']):
                    locs = ', '.join(
                        [self.datas[self.train_idx].labels[i] for i, j in enumerate(r['label_ratios']) if j > 0])
                    print('** This evaluation only represents labels: %s' % locs)

            else:
                # header
                print('__/ Prediction \___')
                print('|  Clf   Coverage |')
                print('| -----  -------- |')

                # baseline
                coverage = self.calc_coverage(predictions['ybkg'])
                print('| {:<5}  {:>8.1%} |'.format('BKG', coverage))

                # clf-inf
                coverage = self.calc_coverage(predictions['yinf'])
                print('| {:<5}  {:>8.1%} |'.format('INF', coverage))

                # clf-match
                coverage = self.calc_coverage(predictions['ymatch'])
                print('| {:<5}  {:>8.1%} |'.format('MATCH', coverage))

                # clf-net
                coverage = self.calc_coverage(predictions['ynet'])
                print('| {:<5}  {:>8.1%} |'.format('NET', coverage))

                # clf-opt
                if 'yopt' in predictions:
                    coverage = self.calc_coverage(predictions['yopt'])
                    print('| {:<5}  {:>8.1%} |'.format('*OPT', coverage))

                print()

        return

    def archive_clfs(self):

        pkg = {
            'inf': self.maxinflidxratio,
            'infhist': self.inflidxratio_history,
            'match': deepcopy(self.unique_link2labels),
            'net': deepcopy(self.clf_net),
        }

        return pkg

    def restore_clfs(self, pkg):

        self.maxinflidxratio = pkg['inf']
        self.inflidxratio_history = pkg['infhist'].copy()
        self.unique_link2labels = deepcopy(pkg['match'])
        self.clf_net = deepcopy(pkg['net'])

        return self

    def blend_data(self, idxs=None):

        if idxs is None:
            idxs = list(range(len(self.datas)))

        datax = Data(verbose=False)
        datax.mimic(self.datas[self.train_idx])
        datax.link2featidx = self.datas[self.train_idx].link2featidx.copy()
        n_nodes = 0
        datax.nidxconvert = dict()
        for i in idxs:
            datax.nodes += self.datas[i].nodes if self.datas[i] else [str(j) for j in range(len(self.datas[i].node_labels))]
            datax.node_labels += self.datas[i].node_labels
            datax.node_links += self.datas[i].node_links
            datax.nidx_train += [k + n_nodes for k in self.datas[i].nidx_train]
            datax.nidx_pred += [k + n_nodes for k in self.datas[i].nidx_pred]
            datax.nidx_exclude += [k + n_nodes for k in self.datas[i].nidx_exclude]
            datax.links = list(set(datax.links) | set(self.datas[i].links))
            datax.nidx2layer += [k + '-%d'%i for k in self.datas[i].nidx2layer] if self.datas[i].nidx2layer and self.train_multilayers else [str(i) for _ in range(len(self.datas[i].node_labels))]
            datax.nidxconvert[i] = {n: n + n_nodes for n in range(len(self.datas[i].node_links))}
            n_nodes = len(datax.node_links)

            # update link frequency
            for l in self.datas[i].links:
                if l in datax.link2freq:
                    datax.link2freq[l] += self.datas[i].link2freq[l]
                else:
                    datax.link2freq[l] = self.datas[i].link2freq[l]

        # re-map layer2nidx
        datax.layer2nidx = dict()
        for n, l in enumerate(datax.nidx2layer):
            if l in datax.layer2nidx:
                datax.layer2nidx[l].append(n)
            else:
                datax.layer2nidx[l] = [n]
        for l, n in datax.layer2nidx.items():
            datax.layer2nidx[l] = set(n)

        datax.link2lidx = {l:j for j,l in enumerate(datax.links)}
        datax.map_data()

        return datax

    def clf_all_predict(self, nidx_target, data=None):

        """
        Predict for the specified indices of the nodes in the Data object with all the base classifiers.

        :param nidx_target: A Python list of indices of nodes in the Data object
        :param data: Optional, Data object loaded with dataset to predict for. If None, then train data is used
        :return: A dictionary containing predictions from all the base classifiers and the final model
        """

        if not data:
            data = self.datas[self.train_idx]

        # Predict with neighbor-inferred clf
        y_inf_pred, inf_predictable = self.clf_inf_predict(nidx_target=nidx_target, data=data)

        # Predict with link-matching clf
        y_match_pred, match_predictable = self.clf_match_predict(nidx_target=nidx_target, data=data)

        # Predict with network-trained clf
        y_net_pred, net_predictable, y_truth = self.clf_net_predict(nidx_target=nidx_target, data=data)

        # Group predictions
        predictions = {
            'nodes': ['/'.join(data.nodes[i]) for i in nidx_target] if data.nodes else [str(i) for i in nidx_target],
            'inf': inf_predictable,
            'match': match_predictable,
            'net': net_predictable,
            'ytruth': y_truth,
            'yinf': y_inf_pred,
            'ymatch': y_match_pred,
            'ynet': y_net_pred
        }

        return predictions

    def clf_inf_predict(self, nidx_target, data=None):

        """
        Predict with the neighbor-infered classifier.

        :param nidx_target: A Python list of indices of nodes in the Data object
        :param data: Optional, Data object loaded with dataset to predict for. If None, then .train_data is used
        :return: 2 variables: a 2-D numpy array of the probabilities of the labels, a numpy array of
        the samples predictable by this classifier
        """

        if not data:
            data = self.datas[self.train_idx]

        y_inferred = []
        predictable = np.zeros(len(nidx_target), dtype=bool)

        for i, n in enumerate(nidx_target):
            nidx = []
            lidx = data.nidx2lidx[n]
            for l in lidx:
                if data.lidx2ratio[l] <= self.maxinflidxratio:
                    nidx += list(data.lidx2nidx[l])
            nidx = set(nidx) & set(data.nidx_train)
            if self.train_multilayers:
                nidx = list((nidx - {n}) & data.layer2nidx[data.nidx2layer[n]])
            else:
                nidx = list(nidx - {n})

            n_nodes = sum([1 for r in nidx if data.node_labels[r]])
            if n_nodes:
                prob_label = np.sum([[1 if l in data.node_labels[r] else 0 for l in data.labels] for r in nidx], axis=0) / n_nodes
                y_inferred.append(prob_label)
                predictable[i] = True
            else:
                y_inferred.append(np.zeros(data.n_labels))

        return np.array(y_inferred), predictable

    def clf_match_predict(self, nidx_target, data=None):

        """
        Predict with the link-matched classifier.

        :param nidx_target: A Python list of indices of nodes in the Data object
        :param data: Optional, Data object loaded with dataset to predict for. If None, then .train_data is used
        :return: 2 variables: a 2-D numpy array of the probabilities of the labels, a numpy array of
        the samples predictable by this classifier
        """

        if not data:
            data = self.datas[self.train_idx]

        pred_locs = []
        predictable = np.zeros(len(nidx_target), dtype=bool)
        for i, n in enumerate(nidx_target):
            links = {data.links[l] for l in data.nidx2lidx[n]}
            locs = []
            for l in links:
                if l in self.unique_link2labels:
                    locs.append(self.unique_link2labels[l])
            if locs:
                locs = set.intersection(*locs)
                if locs:
                    predictable[i] = True

            if locs:
                pred_locs.append(list(locs))
            else:
                pred_locs.append([])

        return self.datas[self.train_idx].encode_labels(pred_locs), predictable

    def clf_net_predict(self, nidx_target, data=None):

        """
        Predict with the network-trained classifier.

        :param nidx_target: A Python list of indices of nodes in the Data object
        :param data: Optional, Data object loaded with dataset to predict for. If None, then train data is used
        :return: 3 variables: a 2-D numpy array of the probabilities of the labels, a numpy array of
        the samples predictable by this classifier, and a numpy array of the true labels
        """

        if not data:
            data = self.datas[self.train_idx]

        X, y, predictable = data.gen_features(nidx_target=nidx_target, perlayer=self.train_multilayers)
        y_pred = self._predict_proba(self.clf_net, X)

        return y_pred, predictable, y

    def bkg_predict(self, n_samples, data=None):

        """
        Predict baseline to evaluate performance.

        :param n_samples: An integer that specifies the number of samples to predict for
        :param data: Optional, Data object loaded with dataset to predict for. If None, then train data is used
        :return: A 2-D numpy array of the probabilities of the labels
        """

        if data is None:
            data = self.datas[self.train_idx]

        y_train = data.gen_labels()
        bkg = DummyClassifier()
        bkg.fit(np.random.rand(len(y_train), 1), y_train)

        return self._predict_proba(bkg, np.random.rand(n_samples, 1))

    @staticmethod
    def _predict_proba(clf, X):

        """
        Mainly for sklearn's classifiers. Since probabilities are needed to compute the ROC, the .predict_proba()
        function is used. However, this function returns both the probabilities to be False [0] and the probabilities to
        be True [1], thus the array is reshaped to only return the probabilities to be True.

        :param clf: An sklearn classifier object
        :param X: A numpy array of the features for a set of data to be predicted with clf
        :return: A numpy array of the probabilities of the labels
        """

        y = clf.predict_proba(X)
        if np.any(np.array(clf.n_classes_) < 2):
            y = np.array([j if i > 1 else np.concatenate((j, np.zeros([len(j), 1])), axis=1) for i, j in zip(clf.n_classes_, y)])

        return np.transpose(y)[1]

    def calc_coverage(self, y):

        """
        Calculate the coverage - the proportion of data that has labels predicted.

        :param y: A numpy array of predictions from any of the classifiers
        :return: A float point variable of the calculated coverage
        """

        coverage = sum(np.any(self._round(y), axis=1)) / len(y)

        return coverage

    @staticmethod
    def _construct_clf_opt_X(predictions):

        """
        This concatentate all the predictions from the base classifiers to create the features used to train/predict
        with the final model.

        :param predictions: The Python dictionary that contains the predictions from the base classifiers
        :return: A numpy array of the concatenated predictions
        """

        return np.concatenate([predictions['yinf'], predictions['ymatch'], predictions['ynet']], axis=1)

    def _train_clfs(self, train_nidxs=None):

        """
        Training the base classifiers with specified samples of the training data. Samples are specified by
        their node indices in the loaded .train_data Data object. If train_nidx is not specified, then all the
        trainable nodes in the .train_data Data object will be used.

        :param train_nidxs: Optional, a list of integers that specified the node indices.
        :return:
        """

        if train_nidxs is None:
            train_nidxs = self.datas[self.train_idx].nidx_train

        X_train, y_train, _ = self.datas[self.train_idx].gen_features(nidx_target=train_nidxs, perlayer=self.train_multilayers)

        if self.verbose:
            print('Train clf-inf: optimizing maxinflidxratio ...')
        self._train_clf_inf(nidx_target=train_nidxs)

        if self.verbose:
            print('Train clf-match: building unique link table ...')
        self.unique_link2labels = self.datas[self.train_idx].gen_match_table(nidx_target=train_nidxs)

        if self.verbose:
            print('Train clf-net: fitting random-forest classifier ...')
        self.clf_net.fit(X_train, y_train)

        return

    def _train_clf_inf(self, nidx_target):

        y0 = self.datas[self.train_idx].gen_labels(nidxs=nidx_target)
        x = minimize(self._opt_maxinflidxratio,
                     np.array([self.maxinflidxratio]),
                     (nidx_target, y0),
                     method='cobyla',
                     options={'rhobeg': 0.1}).x[0]
        self.inflidxratio_history.append(x)
        self.maxinflidxratio = np.median(self.inflidxratio_history)
        if self.verbose:
            print('  maxinflidxratio = %.4f' % self.maxinflidxratio)

        return

    def _opt_maxinflidxratio(self, x, nidx_target, y0):

        self.maxinflidxratio = x

        y, _ = self.clf_inf_predict(nidx_target=nidx_target)

        r = self.scores(y0, y)

        if self.aim:
            score = -r['aucroc'] * r[self.aim.replace('hard', '')]
        else:
            score = -r['aucroc'] * r['f1']

        return score

    def _train_clf_opt(self, predictions):

        """
        Trains the final random forest classifier (aka. the final model) using the ensemble of the predictions from the
        base classifiers.

        :param predictions: A numpy array of the concatenated predictions created with ._construct_clf_opt_X()
        :return:
        """

        X = self._construct_clf_opt_X(predictions)
        y = predictions['ytruth']
        X, y = self._check_train_labels(X, y)

        self.clf_opt.n_estimators += 10
        self.clf_opt.fit(X, y)
        self.clf_opt_trained = True

        if self.aim is not None:
            ypred = self._predict_proba(self.clf_opt, X=X)
            ybkg = self.bkg_predict(n_samples=len(X))
            x = minimize(self._opt_round_cutoff,
                         np.array([self.round_cutoff if self.round_cutoff is not None else 0.5]),
                         (y, ypred, ybkg),
                         method='cobyla',
                         options={'rhobeg': 0.1}).x[0]
            self.round_cutoff_history.append(x)
            self.round_cutoff = np.median(self.round_cutoff_history)
            print('Optimal [%s] round_cutoff=%.4f' % (self.aim, self.round_cutoff))

        return

    def _opt_round_cutoff(self, x, y, ypred, ybkg):

        self.round_cutoff = x

        r = self.scores(y, ypred)
        rbkg = self.scores(y, ybkg)
        c = self.calc_coverage(ypred)

        if 'hard' not in self.aim:
            return -(r[self.aim] / rbkg[self.aim]) * (r['f1'] / rbkg['f1']) * c * np.ceil(x)
        else:
            aim = self.aim.replace('hard', '')
            return -(r[aim] / rbkg[aim]) * c * np.ceil(x)

    @staticmethod
    def _check_train_labels(X, y):

        idx = np.invert(np.any(y == 1, axis=0) & np.any(y == 0, axis=0))
        if np.any(idx):
            X = np.concatenate((X, np.zeros([2, len(X[0])])), axis=0)
            y_ = np.zeros([2, len(y[0])])
            y_[0][idx] = 1
            y = np.concatenate((y, y_), axis=0)

        return X, y

    def scores(self, y, y_pred):

        """
        Compute the performance metrics, which include AUC-ROC, precision, AUC-ROC of each label, precision of
        each label, as well as the proportion of data known to a label (based on the truth).

        :param y: A numpy array of true labels
        :param y_pred: A numpy array of predicted labels by any of the classifiers
        :return: a dict of results
        """

        aucroc = 0.
        precision = 0.
        recall = 0.
        f1 = 0.
        aucroc_labs = np.zeros(self.datas[self.train_idx].n_labels)
        precision_labs = np.zeros(self.datas[self.train_idx].n_labels)
        recall_labs = np.zeros(self.datas[self.train_idx].n_labels)
        f1_labs = np.zeros(self.datas[self.train_idx].n_labels)
        label_ratios = np.mean(y, axis=0)

        if len(y) > 1:
            y_t = np.transpose(y)
            col_keep = np.ones(len(y_t), dtype=bool)
            for i, col_y in enumerate(y_t):
                if 0 not in col_y or 1 not in col_y:
                    col_keep[i] = False

            if sum(col_keep) > 0:
                if not col_keep.all():
                    y = np.transpose(y_t[col_keep])
                    y_pred = np.transpose(np.transpose(y_pred)[col_keep])

                f1 = f1_score(y, self._round(y_pred), average=self.metrics_avg)
                s = f1_score(y, self._round(y_pred), average=None)
                f1_labs[col_keep] = s if sum(col_keep) > 1 else s[1]
                aucroc = roc_auc_score(y, y_pred, average=self.metrics_avg)
                aucroc_labs[col_keep] = roc_auc_score(y, y_pred, average=None)
                precision = precision_score(y, self._round(y_pred), average=self.metrics_avg)
                recall = recall_score(y, self._round(y_pred), average=self.metrics_avg)
                if sum(col_keep) > 1:
                    precision_labs[col_keep] = precision_score(y, self._round(y_pred), average=None)
                    recall_labs[col_keep] = recall_score(y, self._round(y_pred), average=None)
                else:
                    precision_labs[col_keep] = precision_score(y, self._round(y_pred))
                    recall_labs[col_keep] = recall_score(y, self._round(y_pred))
            elif self.verbose:
                print('*Cannot compute other metrics because no label in Truth has alternatives, only precision*')
                precision = precision_score(y, self._round(y_pred), average=self.metrics_avg)
                precision_labs = precision_score(y, self._round(y_pred), average=None)

        elif len(y) == 1:
            if self.verbose:
                print('*Cannot compute other metrics with %d samples, only precision*' % len(y))
                precision = precision_score(y, self._round(y_pred), average=self.metrics_avg)
                precision_labs = precision_score(y, self._round(y_pred), average=None)

        result = {
            'aucroc': aucroc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'aucroc_labs': aucroc_labs,
            'precision_labs': precision_labs,
            'recall_labs': recall_labs,
            'f1_labs': f1_labs,
            'label_ratios': label_ratios
        }

        return result

    def _round(self, y):

        if self.round_cutoff is None:
            return np.round(y)
        else:
            return (y > self.round_cutoff) * 1.

class Data:

    def __init__(self, p_data='', exclude_links=None, labels=None, k_neighbors=3, min_network_size=10,
                 verbose=False, lab_other=True, columns=None, masklayer=None):

        """
        The Data class constructs mappings of nodes to links, generate features for the NET classifier, and
        determine nodes to use to training/evaluation and predictions. All of the data preprocessing is done with
        functions of this class. The data provided to p_data as string of the file path must be a file in the
        tonknows-Data (TDF) format (see tonknows-Data format in tonknows.py for detail).

        :param p_data: Optional, string of the path of the TDF file
        :param exclude_links: Optional, list of string(s) of link names to exclude to building network connections
        :param labels: Optional, list of string(s) of label names to exclude from training
        :param k_neighbors: Optional, integer that indicates the radius of neighorhood to include nodes to generate
        the NET classifier's features
        :param min_network_size: Optional, integer that indicates the minimum number of nodes required to be a
        useful subnetwork, which will be combined with all other subnetwork that possibly exists with the dataset
        :param verbose: Optional, Boolean for whether to verbose
        :param lab_other: Optional, list of label names to be grouped as "others"
        """

        self._header_ = 'Data.'

        self.p_data = p_data
        self.df = pd.DataFrame()
        self.exclude_links = exclude_links if exclude_links else []
        self.labels = labels if labels else []
        self.n_labels = len(labels) if labels else 0
        self.verbose = verbose
        self.lab_other = lab_other
        self.columns = {'links':'links', 'nodes':'nodes', 'labels':'labels', 'layers':'layers'}
        if columns:
            for k, v in columns.items():
                self.columns[k] = v

        self.nodes = list()
        self.node_links = list()
        self.node_labels = list()
        self.links = list()
        self.nidx2lidx = dict()
        self.lidx2nidx = dict()
        self.link2featidx = dict()
        self.lidx2ratio = dict()
        self.link2freq = dict()
        self.link2lidx = dict()
        self.lidx2fidx = dict()

        self.k_neighbors = k_neighbors
        self.min_network_size = min_network_size
        self.maxlidxratio = 0.25
        self.minlinkfreq = 1
        self.spearman_cutoff = 0.1

        self.nidx_train = list()
        self.nidx_pred = list()
        self.nidx_exclude = list()
        self.layer2nidx = dict()
        self.nidx2layer = list()

        self.masklayer = masklayer if masklayer else []
        self._perlayer_ = False

    def build_data(self):

        """
        Perform preprocessing of the dataset.

        :return:
        """

        _header_ = self._header_ + 'build_data(): '

        if self.verbose:
            print(_header_ + 'Building data for %s ...' % self.p_data)

        self.read_data()
        self.map_data()
        self.partition_data()
        self.composition()

        if self.verbose:
            print(_header_ + 'Build complete.')

        return self

    def mimic(self, data):

        self.labels = data.labels.copy()
        self.lab_other = data.lab_other
        self.k_neighbors = data.k_neighbors
        self.min_network_size = data.min_network_size
        self.n_labels = data.n_labels
        self.maxlidxratio = data.maxlidxratio
        self.minlinkfreq = data.minlinkfreq
        self.spearman_cutoff = data.spearman_cutoff

        return self

    def read_data(self, p_data=''):

        """
        Cache the information provided in the TDF file, identify labels, and determine predictable nodes.

        :param p_data: Optional, string of the path of the TDF file.
        :return:
        """

        _header_ = self._header_ + 'read_data(): '

        if p_data:
            self.p_data = p_data

        if not self.p_data:
            raise ValueError(_header_ + 'No data to read.')

        if not os.path.isfile(self.p_data):
            raise FileNotFoundError(_header_ + 'No such file: %s' % self.p_data)

        if self.verbose:
            print(_header_ + 'Reading data from %s ...' % self.p_data)

        if self.nidx_pred:
            # If there are nodes already in .nidx_pred, then they are likely copied over from the train data
            # So, these must be purged prior to reading new data
            print(_header_ + 'Excluding %d predicting nodes transfered from training dataset ...' % len(self.nidx_pred))
            self.nidx_exclude += self.nidx_pred
            self.nidx_pred = []

        # Extract data
        all_links = []
        all_labels = []
        has_other = False
        func = lambda x: [i for i in x.strip().split('/') if i] if isinstance(x, str) else []
        self.df = pd.read_table(self.p_data)
        df = self.df.applymap(func=func)
        has_node = self.columns['nodes'] in df
        has_layer = self.columns['layers'] in df

        for i_row in range(len(df)):
            if has_layer:
                sp = df[self.columns['layers']][i_row][0]
                if sp in self.masklayer:
                    continue
                if sp in self.layer2nidx:
                    self.layer2nidx[sp] |= {i_row}
                else:
                    self.layer2nidx[sp] = {i_row}
                self.nidx2layer.append(sp)
            labs = df[self.columns['labels']][i_row]
            if self.lab_other:
                node_lab = [x if (not self.labels or x in self.labels) else 'other' for x in labs]
                if not has_other and 'other' in node_lab:
                    has_other = True
            else:
                node_lab = [x for x in labs if (not self.labels or x in self.labels)]
            if labs:
                all_labels += labs
                if not node_lab:
                    self.nidx_exclude.append(i_row)
            else:
                self.nidx_pred.append(i_row)
            self.node_links.append([x for x in list(set(df[self.columns['links']][i_row])) if x not in self.exclude_links])
            self.node_labels.append(node_lab)
            if has_node:
                self.nodes.append(df[self.columns['nodes']][i_row])

            all_links += self.node_links[-1]

            # track link frequency
            for link in self.node_links[-1]:
                if link in self.link2freq:
                    self.link2freq[link] += 1
                else:
                    self.link2freq[link] = 1

        self.links += sorted(set(all_links) - set(self.links))
        set_all_labels = set(all_labels)
        if self.labels:
            if self.lab_other and 'other' not in self.labels and has_other:
                self.labels.append('other')

            if self.verbose:
                if self.lab_other:
                    print(_header_ + 'Other labels: %s' % (','.join(set_all_labels - set(self.labels))))
                else:
                    print(_header_ + 'Excluded labels: %s' % (','.join(set_all_labels - set(self.labels))))
        else:
            self.labels = sorted(list(set_all_labels))

        self.n_labels = len(self.labels)

        for idx, link in enumerate(self.links):
            self.link2lidx[link] = idx

        if self.verbose:
            print('  Found %d nodes' % len(self.node_links))
            print('  Found %d links' % len(self.links))

        return self

    def map_data(self):

        """
        Map nodes to links by populating 2 dictionaries: node indices-to-link indices (.nidx2lidx) and
        link indices-to-node indices (.lidx2nidx).

        :return:
        """

        _header_ = self._header_ + 'map_data(): '

        if not self.node_links or not self.links:
            if self.verbose:
                print(_header_ + 'Please read data before mapping.')
            return

        # link statistics
        if self.verbose:
            print(_header_ + 'Computing link statistics ...')
        n_nidx = len(self.node_links)
        link2keep = {}
        n_keep = 0
        for link, freq in self.link2freq.items():
            self.lidx2ratio[self.link2lidx[link]] = freq / n_nidx
            if freq >= self.minlinkfreq:
                link2keep[link] = True
                n_keep += 1
            else:
                link2keep[link] = False
        if self.verbose:
            print(_header_ + 'Using %d / %d links with freq >= %d' % (n_keep, len(self.links), self.minlinkfreq))

        # Mapping
        if self.verbose:
            print(_header_ + 'Mapping data for %s ...' % self.p_data)
        for nidx, links in enumerate(self.node_links):
            lidx = set([self.link2lidx[x] for x in links if x in self.link2lidx and link2keep[x]])
            if lidx:
                self.nidx2lidx[nidx] = lidx
                for c in lidx:
                    if c not in self.lidx2nidx:
                        self.lidx2nidx[c] = {nidx}
                    else:
                        self.lidx2nidx[c] |= {nidx}
            else:
                self.nidx_exclude.append(nidx)
                if nidx in self.nidx_pred:
                    self.nidx_pred.remove(nidx)

        return self

    def partition_data(self):

        """
        Populate the very important .nidx node indices list. nodes in this list are the ones that have
        localization data. They can be used to train and evaluate.

        :return:
        """

        _header_ = self._header_ + 'partition_data(): '

        if self.verbose:
            print(_header_ + 'Partitioning data ...')

        network = self._useful_network()

        if self.nidx_train:
            # The only reason that allows .nidx to not be empty would be that a training Data was copied over
            # hence, the training node indices are retained and need to be excluded
            print(_header_ + 'Excluding %d training nodes transfered from training dataset ...' % len(self.nidx_train))
            nidx = set(self.nidx2lidx.keys()) - set(self.nidx_train)
            self.nidx_exclude += self.nidx_train
            self.nidx_train = []
        else:
            nidx = set(self.nidx2lidx.keys())

        for l in nidx:
            if l in network:
                if self.node_labels[l]:
                    self.nidx_train.append(l)
            else:
                self.nidx_exclude.append(l)

        if self.verbose:
            print(_header_ + 'Found %d nodes' % len(self.nidx2lidx))
            print('  %d nodes with labels of interest' % len(self.nidx_train))
            print('  %d nodes can be used to predict' % len(self.nidx_pred))
            print('  %d nodes cannot be mapped due to lack of mappable links' % len(self.nidx_exclude))

        return self

    def gen_match_table(self, nidx_target=None):

        """
        Construct the link-to-label lookup table for the link-matching classifier.

        :param nidx_target: Optional, list of node indices to construct the table with
        :return: Dictionary of link indices to label names
        """

        _header_ = self._header_ + 'gen_match_table(): '
        if self.verbose:
            print(_header_ + 'Generating link match table ...')

        if nidx_target is None:
            nidx_target = self.nidx_train.copy()

        # Gather all labels for each link
        link_label_table = {}
        for nidx in nidx_target:
            labs = set(self.node_labels[nidx])
            links = {self.links[l] for l in self.nidx2lidx[nidx]}
            for link in links:
                if link in link_label_table:
                    link_label_table[link].append(labs)
                else:
                    link_label_table[link] = [labs]

        # Find the links with consistent labels
        unique_link2labs = {}
        for link, labs in link_label_table.items():
            to_add = True
            if len(labs) > 0:
                shared_labs = set.intersection(*labs)
                if shared_labs:
                    for l in labs:
                        if l != shared_labs:
                            to_add = False
                            break

                    if to_add:
                        unique_link2labs[link] = shared_labs

        return unique_link2labs

    def gen_features(self, nidx_target=None, perlayer=False):

        """
        :param nidx_target: Optional, LIST of node indices to generate features for
        :param perlayer: Optional, Boolean of whether to generate clf-net features using each species' own network
        :return: 3 variables: a 2-D numpy array of nodes-by-features, a 2-D numpy array of nodes-by-labels, and
        a 1-D numpy Boolean array of nodes that can have features generated for (i.e. if yes, then True)
        """

        _header_ = self._header_ + 'gen_features(): '
        if self.verbose:
            print(_header_ + 'Generating features ...')

        self._perlayer_ = perlayer
        if self._perlayer_:
            if not self.layer2nidx:
                self._perlayer_ = False
                if self.verbose:
                    print('  Cannot generate features per layer: No layers field in data')

        if nidx_target is None:
            nidx_target = self.nidx_train.copy()

        if not self.link2featidx:
            self.link2featidx = self.build_link2featidx(nidxs=nidx_target, spearman=True)

        # build link index to feature indices dictionary
        self.lidx2fidx = self.build_lidx2featidx()

        # feature generation
        # **Note** this part is very memory expensive, allocated as much as possible
        with Pool(maxtasksperchild=1) as p:
            r = p.imap(self._gen_feature, nidx_target, chunksize=int(np.ceil(len(nidx_target)/os.cpu_count())))
            X = np.array(list(r))

        # identify predictables
        predictable = np.any(X>0, axis=1)

        return X, self.gen_labels(nidxs=nidx_target), predictable

    def _gen_feature(self, nidx):

        lidx2r = self.build_lidx2radius(nidxs={nidx})
        feat = np.zeros(len(self.link2featidx))
        for l, r in lidx2r.items():
            if l in self.lidx2fidx:
                feat[self.lidx2fidx[l]] = r

        return feat

    def build_link2featidx(self, nidxs=None, spearman=True):

        if self.verbose:
            print('Compiling features ...')

        link2featidx = self._build_link2featidx(nidxs=nidxs, spearman=spearman)

        if self.verbose:
            print('  Compiled %d features' % len(link2featidx))

        return link2featidx

    def _build_link2featidx(self, nidxs=None, spearman=True):

        if nidxs is None:
            nidxs = self.nidx_train

        link2featidx = dict()
        lidxs = []

        # identify all links
        for n in nidxs:
            for l in self.nidx2lidx[n]:
                if l not in lidxs:
                    lidxs.append(l)

        # whether to do Spearman eval
        if spearman:
            # parallelize Spearman eval
            r = self.eval_lidxs(lidxs=lidxs, nidxs=nidxs)

            # screen for valid features
            idx = 0
            for i, x in enumerate(r):
                if x:
                    link2featidx[self.links[lidxs[i]]] = idx
                    idx += 1
        else:
            link2featidx = {self.links[l]:i for i,l in enumerate(lidxs)}

        return link2featidx

    def eval_lidxs(self, lidxs, nidxs=None):

        n_lidxs = len(lidxs)
        lidxs = np.array(lidxs)
        y_feats = np.transpose(self.gen_labels(nidxs=nidxs))
        start = True
        accepted_lidxs = []
        cutoff = self.spearman_cutoff
        r = np.zeros([n_lidxs, self.n_labels], dtype=bool)
        idx = np.ones(n_lidxs, dtype=bool)

        while start or (not np.any(accepted_lidxs) and cutoff <= 0.4):

            if not start and cutoff <= 0.4:
                if self.verbose:
                    print('  Insuffient features, try relaxing Spearman cutoff %.2f -> %.2f' % (cutoff, cutoff * 2))
                cutoff *= 2

            # parallelize Spearman eval
            with Pool(maxtasksperchild=1) as p:
                r[idx] = np.array(list(p.imap(self._eval_lidx,
                                              zip(lidxs[idx], repeat(y_feats), repeat(cutoff), repeat(nidxs)),
                                              chunksize=int(np.ceil(n_lidxs / os.cpu_count())))))

            # check feature coverage
            # coverage0 - whether all labels are covered at least once
            # coverage1 - whether all nodes (nidxs) are covered at least once
            coverage0 = np.any(r, axis=0)
            checked_lidxs = np.any(r, axis=1)
            lidxs_checked = set(lidxs[checked_lidxs])
            coverage1 = [True if self.nidx2lidx[n] & lidxs_checked else False for n in nidxs]
            if np.all(coverage0) and np.all(coverage1):
                accepted_lidxs = checked_lidxs
            else:
                idx[checked_lidxs] = False
                n_lidxs = int(np.sum(idx))
                if not n_lidxs:
                    accepted_lidxs = checked_lidxs

            start = False

        if not np.any(accepted_lidxs):
            if self.verbose:
                print('  Compiling features without Spearman correlation')
            accepted_lidxs = np.ones(n_lidxs, dtype=bool)

        return accepted_lidxs

    def _eval_lidx(self, arg):

        lidx0 = arg[0]
        yfeats = arg[1]
        cutoff = arg[2]
        nidxs = arg[3]

        x = [1 if lidx0 in self.nidx2lidx[n] else 0 for n in nidxs]

        if np.any(x):
            pvals = np.array([spearmanr(x, yi)[1] if np.any(yi) or np.all(yi) else 1. for yi in yfeats])
            pvals[np.isnan(pvals)] = 1.
        else:
            return np.zeros(np.shape(yfeats)[0], dtype=bool)

        return pvals < cutoff

    def build_lidx2featidx(self):

        lidx2fidx = {}
        if self.link2featidx:
            for link, fidx in self.link2featidx.items():
                if link in self.link2lidx:
                    lidx2fidx[self.link2lidx[link]] = fidx
        else:
            raise ValueError('Data.link2featidx cannot be empty when building lidx2featidx dictionary')

        return lidx2fidx

    def build_lidx2radius(self, nidxs, i=0, oldnidxs=None, lidx2r=None):

        if not oldnidxs:
            oldnidxs = set()

        if not lidx2r:
            lidx2r = dict()

        newnidxs = []
        for nidx in nidxs:
            for lidx in self.nidx2lidx[nidx]:
                if lidx not in lidx2r and self.lidx2ratio[lidx] <= self.maxlidxratio:
                    # radius is assigned here
                    lidx2r[lidx] = i+1
                    if i < self.k_neighbors:
                        if self._perlayer_:
                            newnidxs += list(self.lidx2nidx[lidx] & self.layer2nidx[self.nidx2layer[nidx]] - oldnidxs)
                        else:
                            newnidxs += list(self.lidx2nidx[lidx] - oldnidxs)

        oldnidxs |= set(nidxs)

        if newnidxs and i < self.k_neighbors:
            return self.build_lidx2radius(nidxs=set(newnidxs), i=i+1, oldnidxs=(oldnidxs | nidxs), lidx2r=lidx2r)
        else:
            return lidx2r

    def composition(self, nidx=None):

        """
        A breakdown of the composition of the part of the data that have localization labels. This analyzes:
        1. Proportion of data for each label
        2. How many links appear in N-number of nodes

        :param nidx: Optional, LIST of node indices to perform compositional analysis for
        :return:
        """

        if nidx is None:
            nidx = list(self.nidx_train)

        if not nidx:
            if self.verbose:
                print('No node with label to compute composition.')
            return

        y_labels = self.gen_labels(nidxs=nidx)

        if self.verbose:

            _header_ = self._header_ + 'composition(): '

            counts = np.sum(y_labels, axis=0)
            total = len(y_labels)
            ratios = np.array(counts) / total

            print(_header_ + 'Composition by labels (%d):' % self.n_labels)
            for l, c, r in zip(self.labels, counts, ratios):
                print('  %s - %d (%.3f)' % (l, c, r))

            lyrnidx = list(set(nidx) & set(range(len(self.nidx2layer))))
            if lyrnidx:
                layers = {self.nidx2layer[i] for i in lyrnidx}
                print(_header_ + 'Layers found (%d):\n  ' % len(layers) + '\n  '.join(layers))

        return

    def gen_labels(self, nidxs=None, condense_labels=False):

        """
        Generate label labels for each node listed in nidx. The label for each node is a binary array with
        the same shape as the .labels list. Each position corresponds to the label of the same position in the
        .labels list. This is one-hot encoding of the label labels, so 1 is assigned to the position if the
        target node is found in that label, otherwise, assign 0.

        :param nidxs: LIST of node indices
        :param condense_labels: Boolean on whether to just use one representative label
        :return: A numpy array of one-hot encoded labels
        """

        if nidxs is None:
            nidxs = self.nidx_train

        y = []

        for r in nidxs:
            y.append(self.node_labels[r])

        if condense_labels:
            # This should be improved, since this will fail if there are labels with exactly the same number of samples
            # Current solution use a bit of noise to minimize conflicts/favors
            y = self.encode_labels(y)
            lab_weights = 1. - np.mean(y, axis=0)
            noise = np.random.normal(loc=0, scale=0.0001, size=np.shape(y))
            y_condensed = np.argmax(minmax_scale(y * lab_weights + noise, axis=1), axis=1)
            return y_condensed

        return self.encode_labels(y)

    def encode_labels(self, lilabs):

        """
        One-hot encoding of the label names.

        :param lilabs: List of label names generated in .gen_labels()
        :return: A numpy array of one-hot encoded labels
        """

        y = []
        for lab in lilabs:
            y.append([1 if l in lab else 0 for l in self.labels])

        return np.array(y, dtype=float)

    def update_labels(self, nidxs, y):

        """
        Update labels using y array from training/eval/prediction

        :param nidxs:
        :param y:
        :return:
        """

        y = np.array(y, dtype=bool)
        for n, yi in zip(nidxs, y):
            self.node_labels[n] = [self.labels[i] for i, j in enumerate(yi) if j]

        return self

    def recruit(self, nidxs_remain, oldlinks=None):

        """
        Identify which sets of nidxs to evaluate/add and which links to expand in each expansion of recruitment

        :param nidxs_remain:
        :param oldlinks:
        :return:
        """

        steps = [{'nidxs': list(nidxs_remain), 'links': set()}]
        new_links = set()

        if oldlinks is None:
            oldlinks = set(self.link2featidx.keys())

        # all nodes associated to oldlinks
        nidxs = set([n for link in oldlinks if link in self.link2lidx for n in self.lidx2nidx[self.link2lidx[link]]])

        # neighbor nodes to eval/train in this expansion
        new_nidxs = nidxs & nidxs_remain
        if new_nidxs:
            # remaining node for next expansion
            nidxs_remain -= new_nidxs

            # find new links to expand feature set
            steps[0]['nidxs'] = list(new_nidxs)
            new_links = set([self.links[l] for n in nidxs for l in self.nidx2lidx[n]]) - oldlinks
            if new_links:
                steps[0]['links'] = new_links

        if nidxs_remain and new_links:
            return steps + self.recruit(nidxs_remain=nidxs_remain, oldlinks=oldlinks | new_links)

        return steps

    def _compile_networks(self):

        """
        Extract all the networks exist in the dataset. A network is defined as one connected cluster of nodes. There
        can exist discontinuity in the dataset, thus multiple networks can exist. This identifies all of them.

        :return: LIST of LISTs of node indices
        """

        _header_ = self._header_ + '_compile_networks(): '

        if self.verbose:
            print(_header_ + 'Compiling all networks ...')

        networks = []

        all_nidx = set(self.nidx2lidx.keys())

        while all_nidx:

            nidx0 = [all_nidx.pop()]
            network = set(nidx0)

            while nidx0 and all_nidx:

                nidx = set()

                for l in nidx0:
                    lidx = self.nidx2lidx[l]
                    for n in lidx:
                        nidx |= self.lidx2nidx[n]

                nidx -= network
                network |= nidx
                all_nidx -= nidx
                nidx0 = nidx.copy()

            networks.append(network)

        if self.verbose:
            print(_header_ + 'Found %d networks' % len(networks))
            for i, network in enumerate(networks):
                print('  Network %d - %s' % (i, ','.join([str(j) for j in network])))

        return networks

    def _useful_network(self):

        """
        Combine all the networks found by ._compile_networks() that contain at least .min_network_size number of
        nodes (default to 1, i.e. all sub-networks included).

        :return: A LIST of all the included node indices
        """

        networks = self._compile_networks()

        network = []
        for n in networks:
            if len(n) >= self.min_network_size:
                network += list(n)

        return network
