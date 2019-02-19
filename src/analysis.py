#!/usr/bin/env python3

"""
analysis.py

This is for analyzing results. Still needs a lot of work!

"""



# Import
import os
import pandas as pd
import numpy as np
from src.iofunc import open_pkl
from src.model import Model
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

# Classes
class Analysis:

    def __init__(self, p_data=''):

        self.p_data = p_data if p_data and p_data.endswith('/') else p_data + '/'
        self.use_f1 = False

    def compile_batch_train_results(self, d_mods):
        p_mods = [d_mods + x for x in os.listdir(d_mods) if x.endswith('.pkl')]
        data = None

        n_samples = 0

        if p_mods:
            # Get first one
            data, n = self.compile_train_results(p_mod=p_mods[0])
            n_samples += n
            print('Gathered: %s' % p_mods[0])

            # Get the other ones if there are others
            if len(p_mods) > 1:
                for p_mod in p_mods[1:]:
                    tmp, n = self.compile_train_results(p_mod=p_mod)
                    data = data.append(tmp, ignore_index=True)
                    n_samples += n
                    print('Gathered: %s' % p_mod)

        print('\nNumber of samples: %d\n' % n_samples)

        return data

    @staticmethod
    def compile_train_results(p_mod):
        n_samples = 0

        m_pkg, results = open_pkl(p_file=p_mod)

        model = Model()
        model.import_model(m_pkg)
        modelid = model.id.split('-')[-1]
        aim = model.aim if model.aim else 'no-aim'
        filename = os.path.split(p_mod)[1]

        data = {'clf': [],
                'type': [],
                'value': [],
                'kcv': [],
                'irep': [],
                'aim': [],
                'lab': [],
                'modelid': [],
                'file':[],
                }

        layerkey = None
        if model.columns['layers'] in results.columns:
            layerkey = model.columns['layers']
            data['layers'] = []

        clf_code = {'ybkg': 'Baseline',
                    'yinf': 'Inf',
                    'ymatch': 'Match',
                    'ynet': 'Net',
                    'yopt': 'Optimized',
                    }

        for x in ['ybkg', 'yinf', 'ymatch', 'ynet', 'yopt']:

            if x in ['yinf', 'ymatch', 'ynet']:
                if x == 'yinf':
                    idx = results['inf'].values
                elif x == 'ymatch':
                    idx = results['match'].values
                else:
                    idx = results['net'].values
            else:
                # Evaluate opt on all samples
                # this looks silly but easier to just do this given how I stores the arrays in pandas dataframe
                idx = results['net'].values | np.invert(results['net'].values)

            irep = 0
            for i, (y0, y1) in enumerate(zip(results['ytruth'], results[x])):

                r = model.scores(y0[idx[i]], y1[idx[i]])
                predictable = sum(idx[i]) / len(idx[i]) * 100
                predictables = np.sum(y0[idx[i]], axis=0) / len(y0) * 100
                vlayer = results[layerkey][i] if layerkey else None

                if layerkey:
                    # This is a bit more complicated
                    # For each layer, there n-reps on all other layers and an eval on the one-outed layer
                    # The one-outed layer should not count since it didn't participate in training
                    j = i % (model.kfold_cv * model.n_repeat + 1)
                    irep += 1 if j % model.kfold_cv == 0 and j != (model.kfold_cv * model.n_repeat) else 0
                else:
                    irep = i // model.kfold_cv + 1

                # Append AUC-ROC
                data['clf'].append(clf_code[x])
                data['type'].append('aucroc')
                data['value'].append(r['aucroc'])
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['aim'].append(aim)
                data['lab'].append('all')
                if layerkey:
                    data['layers'].append(vlayer)

                # Append F1
                data['clf'].append(clf_code[x])
                data['type'].append('f1')
                data['value'].append(r['f1'])
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['aim'].append(aim)
                data['lab'].append('all')
                if layerkey:
                    data['layers'].append(vlayer)

                # Append Precision
                data['clf'].append(clf_code[x])
                data['type'].append('precision')
                data['value'].append(r['precision'])
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['aim'].append(aim)
                data['lab'].append('all')
                if layerkey:
                    data['layers'].append(vlayer)

                # Append Recall
                data['clf'].append(clf_code[x])
                data['type'].append('recall')
                data['value'].append(r['recall'])
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['aim'].append(aim)
                data['lab'].append('all')
                if layerkey:
                    data['layers'].append(vlayer)

                # Append Coverage
                data['clf'].append(clf_code[x])
                data['type'].append('coverage')
                data['value'].append(model.calc_coverage(y1) * 100)
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['aim'].append(aim)
                data['lab'].append('all')
                if layerkey:
                    data['layers'].append(vlayer)

                # Append Predictable
                data['clf'].append(clf_code[x])
                data['type'].append('predictable')
                data['value'].append(predictable)
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['aim'].append(aim)
                data['lab'].append('all')
                if layerkey:
                    data['layers'].append(vlayer)

                # AUC-ROC per lab
                cov_idx = np.any(y1, axis=1)
                coverages = np.sum(y0[cov_idx], axis=0) / len(y0) * 100
                for j, lab in enumerate(model.datas[model.train_idx].labels):
                    data['clf'].append(clf_code[x])
                    data['type'].append('aucroc_labs')
                    data['value'].append(r['aucroc_labs'][j])
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['aim'].append(aim)
                    data['lab'].append(lab)
                    if layerkey:
                        data['layers'].append(vlayer)

                    data['clf'].append(clf_code[x])
                    data['type'].append('f1_labs')
                    data['value'].append(r['f1_labs'][j])
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['aim'].append(aim)
                    data['lab'].append(lab)
                    if layerkey:
                        data['layers'].append(vlayer)

                    data['clf'].append(clf_code[x])
                    data['type'].append('precision_labs')
                    data['value'].append(r['precision_labs'][j])
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['aim'].append(aim)
                    data['lab'].append(lab)
                    if layerkey:
                        data['layers'].append(vlayer)

                    data['clf'].append(clf_code[x])
                    data['type'].append('recall_labs')
                    data['value'].append(r['recall_labs'][j])
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['aim'].append(aim)
                    data['lab'].append(lab)
                    if layerkey:
                        data['layers'].append(vlayer)

                    data['clf'].append(clf_code[x])
                    data['type'].append('lab_predictable')
                    data['value'].append(predictables[j])
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['aim'].append(aim)
                    data['lab'].append(lab)
                    if layerkey:
                        data['layers'].append(vlayer)

                    data['clf'].append(clf_code[x])
                    data['type'].append('lab_coverage')
                    data['value'].append(coverages[j])
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['aim'].append(aim)
                    data['lab'].append(lab)
                    if layerkey:
                        data['layers'].append(vlayer)

                n_samples += 1

        data['modelid'] = modelid
        data['file'] = filename

        return pd.DataFrame(data), n_samples

    @staticmethod
    def normalize(data, clf='Optimized', bl='Baseline'):
        # This works because all the data for each clf/bkg are loaded in the same order
        # Normalize every metrics to baseline or whatever set as baseline except AUC-ROC
        # Because AUC-ROC already has an absolute baseline for random at 0.5
        clfidx = data['clf'] == clf
        blidx = data['clf'] == bl

        types = ['aucroc', 'f1', 'precision', 'recall']
        dfx = pd.DataFrame()
        for k in types:
            idx = data['type'] == k
            d = data[clfidx & idx]
            if k != 'aucroc':
                d['value'] = data[clfidx & idx]['value'].values - data[blidx & idx]['value'].values
            dfx = dfx.append(d, ignore_index=True)
        dfx = dfx.append(data[clfidx & (data['type'] == 'predictable')], ignore_index=True)
        dfx = dfx.append(data[clfidx & (data['type'] == 'coverage')], ignore_index=True)

        types = ['aucroc_labs', 'f1_labs', 'precision_labs', 'recall_labs']
        for k in types:
            idx = data['type'] == k
            d = data[clfidx & idx]
            if k != 'aucroc_labs':
                d['value'] = data[clfidx & idx]['value'].values - data[blidx & idx]['value'].values
            dfx = dfx.append(d, ignore_index=True)
        dfx = dfx.append(data[clfidx & (data['type'] == 'lab_predictable')], ignore_index=True)
        dfx = dfx.append(data[clfidx & (data['type'] == 'lab_coverage')], ignore_index=True)

        return dfx

    @staticmethod
    def plot00(data, ax, x='clf', y='value', primary=None, secondary=None, ylabels=None, hue=None, title='',
               scale='count', linewidth=1.5, cut=0, hue_order=None, order=None, xlabel=None):

        if not primary:
            primary = ['type', 'aucroc', 0., 1.05]

        if not secondary:
            secondary = ['type', 'predictable', 0., 105]

        if not ylabels:
            ylabels = ['AUC-ROC (violins)', 'Predictables (% of data, bars)']

        sns.violinplot(data=data[data[primary[0]] == primary[1]], x=x, y=y, hue=hue, ax=ax, scale=scale,
                       linewidth=linewidth, cut=cut, hue_order=hue_order, order=order)
        plt.setp(ax.collections, alpha=0.6)
        ax.set_title(title)
        ax.set_ylim([primary[2], primary[3]])
        ax.set_ylabel(ylabels[0])
        ax.set_xlabel(xlabel)
        ax2 = plt.twinx(ax)
        sns.barplot(data=data[data[secondary[0]] == secondary[1]], x=x, y=y, hue=hue, ax=ax2, alpha=0.4,
                    hue_order=hue_order, order=order)
        ax2.set_ylim([secondary[2], secondary[3]])
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax2.patch.set_visible(True)
        ax2.set_ylabel(ylabels[1])
        if ax2.legend_:
            ax2.legend_.remove()

        return

    def plot01(self, data, x='clf', y='value', primary=None, secondary=None, hue=None, width=12, height=7, ylabels=None,
               title='', xlabel='', hue_order=None, order=None, baseline=None, xtickrotate=None):

        if not primary:
            primary = [['type', 'aucroc', 0, 1.05],
                       ['type', 'f1', 0, 1.05],
                       ['type', 'precision', 0, 1.05],
                       ['type', 'recall', 0, 1.05],
                       ]

        if not secondary:
            secondary = [['type', 'predictable', 0, 105],
                         ['type', 'predictable', 0, 105],
                         ['type', 'coverage', 0, 105],
                         ['type', 'coverage', 0, 105],
                         ]

        if not ylabels:
            ylabels = [['AUC-ROC (violins)', 'Predictables (% of data, bars)'],
                       ['F1 (violins)', 'Predictables (% of data, bars)'],
                       ['Precision (violins)', 'Coverage (% of data, bars)'],
                       ['Recall (violins)', 'Coverage (% of data, bars)'],
                       ]

        fig, axs = plt.subplots(2, 2, figsize=(width, height))
        fig.suptitle(title, y=1.05)

        # AUC-ROC
        self.plot00(data, axs[0, 0], primary=primary[0], secondary=secondary[0], ylabels=ylabels[0], x=x, y=y, hue=hue,
               hue_order=hue_order, order=order, xlabel=xlabel)

        # F1
        self.plot00(data, axs[0, 1], primary=primary[1], secondary=secondary[1], ylabels=ylabels[1], x=x, y=y, hue=hue,
               hue_order=hue_order, order=order, xlabel=xlabel)

        # Precision
        self.plot00(data, axs[1, 0], primary=primary[2], secondary=secondary[2], ylabels=ylabels[2], x=x, y=y, hue=hue,
               hue_order=hue_order, order=order, xlabel=xlabel)

        # Recall
        self.plot00(data, axs[1, 1], primary=primary[3], secondary=secondary[3], ylabels=ylabels[3], x=x, y=y, hue=hue,
               hue_order=hue_order, order=order, xlabel=xlabel)

        if hue:
            axs[0, 1].legend(frameon=False, loc='upper left', bbox_to_anchor=(1.2, 1.), ncol=1)
            axs[0, 0].legend_.remove()
            axs[1, 0].legend_.remove()
            axs[1, 1].legend_.remove()

        if baseline:
            if baseline[0] is not None:
                axs[0,0].axhline(y=baseline[0], linestyle='--')
            if baseline[1] is not None:
                axs[0,1].axhline(y=baseline[1])
            if baseline[2] is not None:
                axs[1,0].axhline(y=baseline[2])
            if baseline[3] is not None:
                axs[1,1].axhline(y=baseline[3])

        if xtickrotate:
            axs[0,0].set_xticklabels(axs[0,0].get_xticklabels(), rotation=xtickrotate, ha='right')
            axs[0,1].set_xticklabels(axs[0,1].get_xticklabels(), rotation=xtickrotate, ha='right')
            axs[1,0].set_xticklabels(axs[1,0].get_xticklabels(), rotation=xtickrotate, ha='right')
            axs[1,1].set_xticklabels(axs[1,1].get_xticklabels(), rotation=xtickrotate, ha='right')

        fig.tight_layout(w_pad=1)

        plt.draw()

        return

    def plot02(self, data, x='clf', y='value', primary=None, secondary=None, hue=None, width=12, height=7, ylabels=None,
               title='', xlabel='', hue_order=None, baseline=None, xtickrotate=None, ylog=False):

        if not primary:
            primary = [['type', 'aucroc', 0, 1.05],
                       ['type', 'f1', 0, 1.05],
                       ['type', 'precision', 0, 1.05],
                       ['type', 'recall', 0, 1.05],
                       ]

        if not secondary:
            secondary = [['type', 'predictable', 0, 105],
                         ['type', 'predictable', 0, 105],
                         ['type', 'coverage', 0, 105],
                         ['type', 'coverage', 0, 105],
                         ]

        if not ylabels:
            ylabels = [['AUC-ROC (violins)', 'Predictables (% of data, bars)'],
                       ['F1 (violins)', 'Predictables (% of data, bars)'],
                       ['Precision (violins)', 'Coverage (% of data, bars)'],
                       ['Recall (violins)', 'Coverage (% of data, bars)'],
                       ]

        fig, axs = plt.subplots(1, 4, figsize=(width, height))
        fig.suptitle(title, y=1.05)

        # AUC-ROC
        self.plot00(data, axs[0], primary=primary[0], secondary=secondary[0], ylabels=ylabels[0], x=x, y=y, hue=hue,
               hue_order=hue_order, xlabel=xlabel)

        # F1
        self.plot00(data, axs[1], primary=primary[1], secondary=secondary[1], ylabels=ylabels[1], x=x, y=y, hue=hue,
               hue_order=hue_order, xlabel=xlabel)

        # Precision
        self.plot00(data, axs[2], primary=primary[2], secondary=secondary[2], ylabels=ylabels[2], x=x, y=y, hue=hue,
               hue_order=hue_order, xlabel=xlabel)

        # Recall
        self.plot00(data, axs[3], primary=primary[3], secondary=secondary[3], ylabels=ylabels[3], x=x, y=y, hue=hue,
               hue_order=hue_order, xlabel=xlabel)

        if hue:
            axs[3].legend(frameon=False, loc='upper left', bbox_to_anchor=(1.2, 1.), ncol=1)
            axs[0].legend_.remove()
            axs[1].legend_.remove()
            axs[2].legend_.remove()

        if baseline:
            if baseline[0] is not None:
                axs[0].axhline(y=baseline[0], linestyle='--')
            if baseline[1] is not None:
                axs[1].axhline(y=baseline[1])
            if baseline[2] is not None:
                axs[2].axhline(y=baseline[2])
            if baseline[3] is not None:
                axs[3].axhline(y=baseline[3])

        if xtickrotate:
            axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=xtickrotate, ha='right')
            axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=xtickrotate, ha='right')
            axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=xtickrotate, ha='right')
            axs[3].set_xticklabels(axs[3].get_xticklabels(), rotation=xtickrotate, ha='right')

        if ylog:
            axs[1].set(yscale='log')
            axs[2].set(yscale='log')
            axs[3].set(yscale='log')

        fig.tight_layout(w_pad=1)

        plt.draw()

        return

    @staticmethod
    def step_precisions(d_out, model, data, predictions, evaluations=None):

        if not os.path.isdir(d_out):
            os.makedirs(d_out)
            print('Created: %s' % ''.join(os.path.realpath(d_out)))

        # Write predictions at incremental precision
        for cutoff in np.linspace(0., 0.5, 11):
            p_out = d_out + 'precision_%d.tsv' % (cutoff * 100)
            model.round_cutoff = cutoff
            model.write_predictions(predictions, evaluations, data=data, p_out=p_out)
