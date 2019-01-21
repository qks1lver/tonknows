#!/usr/bin/env python3

"""
analysis.py
"""



# Import
import os
import re
import pandas as pd
import numpy as np
from src.iofunc import open_pkl
from src.model import Model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import MultiLabelBinarizer, minmax_scale
from sklearn.cluster import DBSCAN, KMeans
import pdb

# Classes
class Analysis:

    def __init__(self, p_data='', dir_metdomains='', dir_pmn=''):

        self.p_data = p_data if p_data.endswith('/') else p_data + '/'
        self.dir_metdomains = dir_metdomains if dir_metdomains.endswith('/') else dir_metdomains + '/'
        self.dir_pmn = dir_pmn if dir_pmn.endswith('/') else dir_pmn + '/'

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

        primary = [['type', 'loc_aucroc', 1.05], ['type', 'loc_precision', 1.05]]
        secondary = [['type', 'loc_predictable', 105], ['type', 'loc_coverage', 105]]

        self.plot01(data=data, x='clf', xlabel='Classifiers', width=6, height=7, title='Classifier Performances', ylabels=ylabels)
        self.plot01(data=data, primary=primary, secondary=secondary, x='loc', xlabel='Compartments', width=12,
                    title='Performance per compartment', hue='clf', ylabels=ylabels)

        if current_kcv and current_nrep:
            self.plot01(data=data[(data['kcv'] == current_kcv) & (data['irep'] == current_nrep)], x='clf',
                        xlabel='Classifiers', ylabels=ylabels, width=6, height=7,
                        title='Classifier Performances - %d reps, %d CVs' % (current_nrep, current_kcv))
            self.plot01(data=data[(data['kcv'] == current_kcv) & (data['irep'] == current_nrep)], primary=primary,
                        secondary=secondary, x='loc', xlabel='Compartments', ylabels=ylabels, width=12, hue='clf',
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
                'loc':[],
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

                r = model._scores(y0[idx[i]], y1[idx[i]])
                predictable = sum(idx[i]) / len(idx[i]) * 100
                predictables = np.sum(y0[idx[i]], axis=0) / len(y0) * 100

                irep = i // model.kfold_cv + 1

                # Append AUC-ROC
                data['clf'].append(clf_code[x])
                data['type'].append('aucroc')
                data['value'].append(r['aucroc'] if self.use_f1 else r['f1'])
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['loc'].append('all')
                data['modelid'].append(model.id)

                # Append Precision
                data['clf'].append(clf_code[x])
                data['type'].append('precision')
                data['value'].append(r['precision'])
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['loc'].append('all')
                data['modelid'].append(model.id)

                # Append Coverage
                data['clf'].append(clf_code[x])
                data['type'].append('coverage')
                data['value'].append(model._calc_coverage(y1) * 100)
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['loc'].append('all')
                data['modelid'].append(model.id)

                # Append Predictable
                data['clf'].append(clf_code[x])
                data['type'].append('predictable')
                data['value'].append(predictable)
                data['kcv'].append(model.kfold_cv)
                data['irep'].append(irep)
                data['loc'].append('all')
                data['modelid'].append(model.id)

                # AUC-ROC per loc
                cov_idx = np.any(y1, axis=1)
                coverages = np.sum(y0[cov_idx], axis=0) / len(y0) * 100
                for j, (la, lp, loc) in enumerate(zip(r['aucroc_locs'] if self.use_f1 else r['f1_locs'], r['precision_locs'], model.labels)):
                    data['clf'].append(clf_code[x])
                    data['type'].append('loc_aucroc')
                    data['value'].append(la)
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['loc'].append(loc)
                    data['modelid'].append(model.id)

                    data['clf'].append(clf_code[x])
                    data['type'].append('loc_precision')
                    data['value'].append(lp)
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['loc'].append(loc)
                    data['modelid'].append(model.id)

                    data['clf'].append(clf_code[x])
                    data['type'].append('loc_predictable')
                    data['value'].append(predictables[j])
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['loc'].append(loc)
                    data['modelid'].append(model.id)

                    data['clf'].append(clf_code[x])
                    data['type'].append('loc_coverage')
                    data['value'].append(coverages[j])
                    data['kcv'].append(model.kfold_cv)
                    data['irep'].append(irep)
                    data['loc'].append(loc)
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

    def pathway_localization(self, angle='domains'):

        if not self.p_data:
            print('\n  { No data source for pathway localization analysis, .data is empty }\n')
            return None

        p_files = [self.p_data + p for p in os.listdir(self.p_data) if p.endswith('.tsv')]
        if len(p_files) == 0:
            print('\n  { No tables (.tsv) under %s }' % self.p_data)
            return None
        else:
            print('\n  -- Found %d tables --\n' % len(p_files))

        data = {}
        for p_file in p_files:
            print('Reading: %s' % p_file)
            df = pd.read_table(p_file)
            if 'pathways' not in df:
                print('No "pathways" field in file: %s' % p_file)
            else:
                species = os.path.split(p_file)[-1].replace('.tsv', '')
                if angle == 'species':
                    # cache pathway-localization for each species
                    pathway_locs = self._compile_pathway_locs(df)
                    data[species] = pathway_locs
                elif angle == 'pathway':
                    # cache species-localization for each pathway
                    for k, v in self._compile_species_locs(df, species).items():
                        if k in data:
                            data[k] |= v
                        else:
                            data[k] = v
                elif angle == 'pathwayrates':
                    for k, v in self._compile_pathway2loc(df=df).items():
                        if k in data:
                            data[k] += v
                        else:
                            data[k] = v
                elif angle == 'domains':
                    data[species] = self._compile_gids2loc(df=df)
                else:
                    data = {}

        if angle == 'pathwayrates':
            # Compute frequency of each pathway
            all_locs = sorted(set([l for _,g in data.items() for l in g]))
            X = np.zeros([len(data), len(all_locs)])
            for i,k in enumerate(data):
                for l in data[k]:
                    X[i][all_locs.index(l)] += 1

            # Scale
            X = minmax_scale(X, axis=1)

            # Cluster
            Xt = self._cluster(X)
            # TODO

        elif angle == 'domains':
            # Compare by metabolic domains

            # Check pre-reqs
            if not self.dir_metdomains:
                print('  { Missing metabolic domain files directory path }')
                return None

            if not self.dir_pmn:
                print('  { Missing PMN directory path }')
                return None

            # Identify compartments
            print('\nIdentifying compartments ...')
            all_locs = sorted(set([l for g2l in data.values() for locs in g2l.values() for l in locs]))
            n_all_locs = len(all_locs)
            print('Found %d compartments: %s' % (n_all_locs, ', '.join(all_locs)))

            # Species-to-metabolic-domains
            print('\nGathering metabolic domains for each species ...')
            sp2doms = self.all_metabolic_domains(p_dir=self.dir_metdomains)

            # Identify domains
            print('\nIdentifying domains ...')
            all_doms = sorted(set([d for g2d in sp2doms.values() for doms in g2d.values() for d in doms]))
            n_all_doms = len(all_doms)
            print('Found %d domains: %s' % (n_all_doms, ', '.join(all_doms)))

            # Transform data
            print('\nTransforming data ...')
            X = {'Species':[],
                 'Domain':[],
                 'Compartment':[],
                 'value':[]}

            for sp, g2l in data.items():
                x = np.zeros([n_all_doms, n_all_locs])
                if sp in sp2doms:
                    g2d = sp2doms[sp]
                    for g, locs in g2l.items():
                        gids = g.split(',')
                        for gid in gids:
                            if gid in g2d:
                                for d in g2d[gid]:
                                    for l in locs:
                                        x[all_doms.index(d), all_locs.index(l)] += 1
                                # Because gene-IDs separated by comma are actually referring to the same entry, we only
                                # want to count an entry once, thus if one of the gene-IDs in the entry is found then
                                # this entry is closed (exits)
                                break

                    x = minmax_scale(x, axis=1) * 100

                    for i, d in enumerate(x):
                        for j, l in enumerate(d):
                            X['Species'].append(sp)
                            X['Domain'].append(all_doms[i])
                            X['Compartment'].append(all_locs[j][:3])
                            X['value'].append(l)

                else:
                    print('  { %s not found in sp2doms }' % sp)

            df = pd.DataFrame(X)
            self._catplot(df)

        else:
            # ignore shared pathway-locs
            total = set()
            for _, x in data.items():
                total |= x
            print('Total %d combinations' % len(total))

            shared = total.copy()
            for _, x in data.items():
                shared &= x
            print('Species share %d combinations' % len(shared))

            diff = total - shared
            e = MultiLabelBinarizer().fit([diff])
            print('Using %d semi-unique combinations' % len(diff))

            # Binarize
            X = e.fit_transform([set(x) for _,x in data.items()])
            sns.heatmap(X)

        plt.show()

        return

    @staticmethod
    def _catplot(df, x='Compartment', y='value', sub='Domain', hue='Species', ylabels='% genes', ncols=3, width=8, height=10):

        sns.set(style="whitegrid", context='paper', font_scale=1)

        subs = df[sub].unique()
        nsubs = len(subs)
        nrows = int(np.ceil(nsubs / ncols))

        nlegend = len(df[hue].unique())

        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(width, height))

        isub = 0
        for i in range(nrows):
            for j in range(ncols):
                if isub < nsubs:
                    sns.barplot(ax=axs[i][j], data=df[df[sub] == subs[isub]], x=x, y=y, hue=hue)
                    if j > 0 or i > 0:
                        axs[i][j].legend_.remove()
                    else:
                        axs[i][j].legend(frameon=False, loc='upper left', bbox_to_anchor=(0, 1.8), ncol=nlegend)
                    axs[i][j].set_title(subs[isub])
                    axs[i][j].set_ylabel(ylabels)
                    axs[i][j].set_xlabel(x)
                else:
                    break
                isub += 1

        fig.tight_layout(w_pad=1)
        plt.draw()

        return

    @staticmethod
    def _cluster(X, n_cluster0=10):

        print('Clustering ...')

        Xt = np.zeros(len(X))
        for i in range(n_cluster0)[::-1]:
            m = KMeans(n_clusters=(i+1), n_init=25)
            Xt = m.fit_predict(X)
            n_bads = len([j for j in range(max(Xt) + 1) if len(np.where(Xt == j)[0]) < 2])
            if n_bads == 0:
                print('  Ideal clustering found at n_cluster=%d' % (i+1))
                break

        return Xt

    @staticmethod
    def _compile_pathway_locs(df):

        y = []
        for i in range(len(df)):
            if not pd.isna(df['known_localizations'][i]) and not pd.isna(df['pathways'][i]):
                for p in df['pathways'][i].split('/'):
                    for l in df['known_localizations'][i].split('/'):
                        y.append('%s - %s' % (p, l))
            elif not pd.isna(df['predictions'][i]) and not pd.isna(df['pathways'][i]):
                for p in df['pathways'][i].split('/'):
                    for l in df['predictions'][i].split('/'):
                        y.append('%s - %s' % (p, l))

        return set(y)

    @staticmethod
    def _compile_pathway2loc(df, ignore_other=True):

        y = {}
        for i in range(len(df)):
            if not pd.isna(df['known_localizations'][i]) and not pd.isna(df['pathways'][i]):
                for p in df['pathways'][i].split('/'):
                    for l in df['known_localizations'][i].split('/'):
                        if ignore_other and l == 'other':
                            continue
                        if p in y:
                            y[p].append(l)
                        else:
                            y[p] = [l]
            elif not pd.isna(df['predictions'][i]) and not pd.isna(df['pathways'][i]):
                for p in df['pathways'][i].split('/'):
                    for l in df['predictions'][i].split('/'):
                        if ignore_other and l == 'other':
                            continue
                        if p in y:
                            y[p].append(l)
                        else:
                            y[p] = [l]

        return y

    @staticmethod
    def _compile_gids2loc(df, ignore_other=True):

        y = {}
        for i in range(len(df)):
            if not pd.isna(df['known_localizations'][i]) and not pd.isna(df['geneid'][i]):
                for g in df['geneid'][i].split('/'):
                    g = g.split('.')[0]
                    for l in df['known_localizations'][i].split('/'):
                        if ignore_other and l == 'other':
                            continue
                        if g in y:
                            y[g] |= {l}
                        else:
                            y[g] = {l}
            elif not pd.isna(df['predictions'][i]) and not pd.isna(df['geneid'][i]):
                for g in df['geneid'][i].split('/'):
                    g = g.split('.')[0]
                    for l in df['predictions'][i].split('/'):
                        if ignore_other and l == 'other':
                            continue
                        if g in y:
                            y[g] |= {l}
                        else:
                            y[g] = {l}

        return y

    @staticmethod
    def _compile_species_locs(df, species, ignore_other=True):

        y = {}
        for i in range(len(df)):
            if not pd.isna(df['known_localizations'][i]) and not pd.isna(df['pathways'][i]):
                for p in df['pathways'][i].split('/'):
                    for l in df['known_localizations'][i].split('/'):
                        if ignore_other and l == 'other':
                            continue
                        if p in y:
                            y[p] |= {'%s - %s' % (species, l)}
                        else:
                            y[p] = {'%s - %s' % (species, l)}
            elif not pd.isna(df['predictions'][i]) and not pd.isna(df['pathways'][i]):
                for p in df['pathways'][i].split('/'):
                    for l in df['predictions'][i].split('/'):
                        if ignore_other and l == 'other':
                            continue
                        if p in y:
                            y[p] |= {'%s - %s' % (species, l)}
                        else:
                            y[p] = {'%s - %s' % (species, l)}

        return y

    @staticmethod
    def add_pathway(p_prediction='', p_pathway_dat=''):

        # Cache rxn-to-pathways
        rxn2pathway = {}
        with open(p_pathway_dat, 'r', encoding='ISO-8859-1') as f:
            pathwayid = ''
            for l in f:
                if l.startswith('UNIQUE-ID'):
                    pathwayid = l.strip().split(' - ')[1]
                if pathwayid and l.startswith('REACTION-LIST'):
                    rxn = l.strip().split(' - ')[1]
                    if rxn in rxn2pathway:
                        rxn2pathway[rxn] += '/' + pathwayid
                    else:
                        rxn2pathway[rxn] = pathwayid

        # add pathway column based on rxn ID
        df = pd.read_table(p_prediction)
        pathways = []
        n_found = 0
        for rxn in df['reaction_ids'].values:
            if rxn in rxn2pathway:
                pathways.append(rxn2pathway[rxn])
                n_found += 1
            else:
                pathways.append('')
        print('Found pathways for %d / %d reactions' % (n_found, len(df)))
        df['pathways'] = pathways
        df.to_csv(path_or_buf=p_prediction, sep='\t')

        return

    @staticmethod
    def _cache_metabolic_domain(p_data):

        gids = []
        mapping = []
        with open(p_data, 'r') as f:
            domains = np.array(f.readline().strip().split('\t')[1:])
            for l in f:
                tmp = l.strip().split('\t')
                gids.append(tmp[0])
                mapping.append([int(i) for i in tmp[1:]])

        mapping = np.array(mapping, dtype=bool)
        data = {}
        for g, m in zip(gids, mapping):
            if g in data:
                data[g] |= set(domains[m])
            else:
                data[g] = set(domains[m])

        return data

    def all_metabolic_domains(self, p_dir):

        if not p_dir.endswith('/'):
            p_dir += '/'

        mapping = {}
        if os.path.isfile(p_dir + 'name_mapping.txt'):
            with open(p_dir + 'name_mapping.txt', 'r') as f:
                for l in f:
                    tmp = l.strip().split(':')
                    mapping[tmp[0]] = tmp[1]

        p_files = [p_dir + p for p in os.listdir(p_dir) if 'Metabolic_Domains' in p]
        print('Found %d files' % len(p_files))

        all_data = {}
        exp = re.compile(r'mains_(.+?).txt')
        for p_file in p_files:
            species = exp.findall(p_file)[0]
            print('Caching %s ...' % species)

            if mapping and species in mapping:
                species = mapping[species]

            all_data[species] = self._cache_metabolic_domain(p_file)

        return all_data

    @staticmethod
    def _cache_gene2gid(p_data):

        y = {}
        with open(p_data, 'r') as f:
            gid = ''
            genes = []
            for l in f:
                if l.startswith('UNIQUE-ID'):
                    if gid and genes:
                        for g in genes:
                            y[g] = gid
                    gid = l.strip().split(' - ')[1]
                    genes = []
                elif l.startswith('ACCESSION-1'):
                    genes.append(l.strip().split(' - ')[1])

        return y

    def all_gids(self, p_dir):

        if not p_dir.endswith('/'):
            p_dir += '/'

        p_files = {species:p_dir + species + '/genes.dat' for species in os.listdir(p_dir) if os.path.isfile(p_dir + species + '/genes.dat')}
        print('Found %d files' % len(p_files))

        ys = {}
        for sp, p_file in p_files.items():
            print('Caching %s ...' % sp)
            ys[sp] = self._cache_gene2gid(p_data=p_file)

        return ys

    @staticmethod
    def _parse_gene(species, gene):

        if species == 'tomato':
            tmp = gene.strip().split('.')
            if len(tmp) > 2:
                return '.'.join(tmp[:2])
            else:
                return gene
        elif species == 'maize':
            tmp = gene.strip().split('_')
            if len(tmp) > 1:
                return tmp[0]
            else:
                return gene
        else:
            tmp = gene.strip().split('.')
            if len(tmp) > 1:
                return tmp[0]
            else:
                return gene
