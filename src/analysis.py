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

    def compile_batch_train_results(self):

        p_mods = [self.p_data + x for x in os.listdir(self.p_data) if x.endswith('.pkl')]
        data = None

        current_nrep = None
        current_kcv = None
        n_samples = 0

        if p_mods:
            data, n = self.compile_train_results(p_file=p_mods[0])
            n_samples += n
            print('Gathered: %s' % p_mods[0])
            if len(p_mods) > 1:
                for p_mod in p_mods[1:]:
                    if 'current' not in p_mod:
                        tmp, n = self.compile_train_results(p_file=p_mod)
                        data = data.append(tmp, ignore_index=True)
                        n_samples += n
                        print('Gathered: %s' % p_mod)
                    else:
                        mpkg, _ = open_pkl(p_file=p_mod)
                        m = Model()
                        m.import_model(mpkg)
                        current_nrep = m.n_repeat
                        current_kcv = m.kfold_cv

        print('\nNumber of samples: %d\n' % n_samples)

        self.plot_train_results(data=data, current_kcv=current_kcv, current_nrep=current_nrep)

        return data

    def plot_train_results(self, data, current_kcv=None, current_nrep=None):

        if self.use_f1:
            ylabels = [['F1 (violins)', 'Predictables (% of data, bars)'], ['Precision (violins)', 'Coverage (% of data, bars)']]
        else:
            ylabels = None

        primary = [['type', 'lab_aucroc', 1.05], ['type', 'lab_precision', 1.05]]
        secondary = [['type', 'lab_predictable', 105], ['type', 'lab_coverage', 105]]

        self.plot01(data=data, x='clf', xlabel='Classifiers', width=6, height=7, title='Classifier Performances', ylabels=ylabels)
        self.plot01(data=data, primary=primary, secondary=secondary, x='lab', xlabel='Compartments', width=12,
                    title='Performance per compartment', hue='clf', ylabels=ylabels)

        if current_kcv and current_nrep:
            self.plot01(data=data[(data['kcv'] == current_kcv) & (data['irep'] == current_nrep)], x='clf',
                        xlabel='Classifiers', ylabels=ylabels, width=6, height=7,
                        title='Classifier Performances - %d reps, %d CVs' % (current_nrep, current_kcv))
            self.plot01(data=data[(data['kcv'] == current_kcv) & (data['irep'] == current_nrep)], primary=primary,
                        secondary=secondary, x='lab', xlabel='Compartments', ylabels=ylabels, width=12, hue='clf',
                        title='Performance per compartment - %d reps, %d CVs' % (current_nrep, current_kcv))
        self.plot01(data=data, x='clf', xlabel='Classifiers', width=12, hue='kcv', title='K-CVs on Performance')
        self.plot01(data=data[data['irep'] < 7], x='clf', xlabel='Classifiers', ylabels=ylabels, width=16, hue='irep',
                    title='N-repeats of warm random forest on Performance')
        self.plot01(data=data, x='clf', xlabel='Classifiers', ylabels=ylabels, width=16, hue='modelid',
                    title='All models')

        plt.show()

        return

    def compile_train_results(self, p_file):

        n_samples = 0

        m_pkg, results = open_pkl(p_file=p_file)

        model = Model()
        model.import_model(m_pkg)

        data = {'clf':[],
                'type':[],
                'value':[],
                'kcv':[],
                'irep':[],
                'lab':[],
                'modelid':[]}

        clf_code = {'ybkg': 'Baseline',
                    'yinf': 'Inf',
                    'ymatch': 'Match',
                    'ynet': 'Net',
                    'yopt': 'Optimized'}

        for x in ['ybkg', 'yinf', 'ymatch', 'ynet', 'yopt']:

            if x in ['yinf', 'ymatch', 'ynet']:
                if x == 'yinf':
                    idx = results['inf'].values
                elif x == 'ymatch':
                    idx = results['match'].values
                else:
                    idx = results['net'].values
            else:
                idx = results['net'].values | np.invert(results['net'].values)

            for i, (y0, y1) in enumerate(zip(results['ytruth'], results[x])):

                r = model.scores(y0[idx[i]], y1[idx[i]])
                predictable = sum(idx[i]) / len(idx[i]) * 100
                predictables = np.sum(y0[idx[i]], axis=0) / len(y0) * 100

                irep = i // model.kfold_cv + 1

                # Append AUC-ROC
                data['clf'].append(clf_code[x])
                data['type'].append('aucroc')
                data['value'].append(r['aucroc'] if self.use_f1 else r['f1'])
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['lab'].append('all')
                data['modelid'].append(model.id)

                # Append Precision
                data['clf'].append(clf_code[x])
                data['type'].append('precision')
                data['value'].append(r['precision'])
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['lab'].append('all')
                data['modelid'].append(model.id)

                # Append Coverage
                data['clf'].append(clf_code[x])
                data['type'].append('coverage')
                data['value'].append(model.calc_coverage(y1) * 100)
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['lab'].append('all')
                data['modelid'].append(model.id)

                # Append Predictable
                data['clf'].append(clf_code[x])
                data['type'].append('predictable')
                data['value'].append(predictable)
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['lab'].append('all')
                data['modelid'].append(model.id)

                # AUC-ROC per lab
                cov_idx = np.any(y1, axis=1)
                coverages = np.sum(y0[cov_idx], axis=0) / len(y0) * 100
                for j, (la, lp, lab) in enumerate(zip(r['aucroc_labs'] if self.use_f1 else r['f1_labs'], r['precision_labs'], model.labels)):
                    data['clf'].append(clf_code[x])
                    data['type'].append('lab_aucroc')
                    data['value'].append(la)
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['lab'].append(lab)
                    data['modelid'].append(model.id)

                    data['clf'].append(clf_code[x])
                    data['type'].append('lab_precision')
                    data['value'].append(lp)
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['lab'].append(lab)
                    data['modelid'].append(model.id)

                    data['clf'].append(clf_code[x])
                    data['type'].append('lab_predictable')
                    data['value'].append(predictables[j])
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['lab'].append(lab)
                    data['modelid'].append(model.id)

                    data['clf'].append(clf_code[x])
                    data['type'].append('lab_coverage')
                    data['value'].append(coverages[j])
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['lab'].append(lab)
                    data['modelid'].append(model.id)

                n_samples += 1

        return pd.DataFrame(data), n_samples

    @staticmethod
    def plot01(data, x='clf', y='value', primary=None, secondary=None, hue=None, width=8, height=8, ylabels=None, title='', xlabel=''):

        if not primary:
            primary = [['type', 'aucroc', 1.05], ['type', 'precision', 1.05]]

        if not secondary:
            secondary = [['type', 'predictable', 105], ['type', 'coverage', 105]]

        if not ylabels:
            ylabels = [['AUC-ROC (violins)', 'Predictables (% of data, bars)'], ['Precision (violins)', 'Coverage (% of data, bars)']]

        sns.set(style="whitegrid", context='paper', font_scale=1.4)

        fig, axs = plt.subplots(2, figsize=(width, height))

        # AUC-ROC
        sns.violinplot(data=data[data[primary[0][0]] == primary[0][1]], x=x, y=y, hue=hue, ax=axs[0], scale='count', linewidth=1.5, cut=0)
        plt.setp(axs[0].collections, alpha=0.6)
        axs[0].set_title(title)
        axs[0].set_ylim([0, primary[0][2]])
        axs[0].set_ylabel(ylabels[0][0])
        axs[0].set_xlabel('')
        ax2 = plt.twinx(axs[0])
        sns.barplot(data=data[data[secondary[0][0]] == secondary[0][1]], x=x, y=y, hue=hue, ax=ax2, alpha=0.4)
        ax2.set_ylim([0, secondary[0][2]])
        axs[0].set_zorder(ax2.get_zorder() + 1)
        axs[0].patch.set_visible(False)
        ax2.patch.set_visible(True)
        ax2.set_ylabel(ylabels[0][1])
        if ax2.legend_:
            ax2.legend_.remove()

        # Precision
        sns.violinplot(data=data[data[primary[1][0]] == primary[1][1]], x=x, y=y, hue=hue, ax=axs[1], scale='count', linewidth=1.5, cut=0)
        plt.setp(axs[1].collections, alpha=0.6)
        axs[1].set_ylim([0, primary[1][2]])
        axs[1].set_ylabel(ylabels[1][0])
        ax3 = plt.twinx(axs[1])
        sns.barplot(data=data[data[secondary[1][0]] == secondary[1][1]], x=x, y=y, hue=hue, ax=ax3, alpha=0.4)
        ax3.set_ylim([0, secondary[1][2]])
        axs[1].set_zorder(ax3.get_zorder() + 1)
        axs[1].patch.set_visible(False)
        ax3.patch.set_visible(True)
        ax3.set_ylabel(ylabels[1][1])
        if ax3.legend_:
            ax3.legend_.remove()
        if xlabel:
            axs[1].set_xlabel(xlabel)

        axs[0].title.set_y(1.15)
        if hue:
            ncol = len(data[hue].unique())
            axs[0].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=ncol)
            if axs[1].legend_:
                axs[1].legend_.remove()

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
