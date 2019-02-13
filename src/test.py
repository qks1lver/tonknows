#!/usr/bin/env python3

# Imports
import numpy as np
import pdb

# Functions

# Classes
class Dummy:

    def __init__(self, n_labels=5, n_links=500, n_nodes=500):

        self.n_labels = n_labels
        self.labels = ['lab-%d' % i for i in range(self.n_labels)]

        self.n_links = n_links
        self.links = ['l-%d' % i for i in range(self.n_links)]

        self.n_nodes = n_nodes
        self.nodes = ['n-%d' % i for i in range(self.n_nodes)]

        self.linkrange = [2,5]

        self.node2labels = {}
        self.node2links = {}
        self.link2nodes = {}
        self.link2labels = {}
        self.label2links = {}

    def init_networks(self):

        """
        This initialize the purest network with n-subnetworks for the n-labels

        :return:
        """

        self.populate_link2labels()
        self.map_label2links()
        self.populate_node2links()
        self.map_link2nodes()

        return self

    def populate_link2labels(self, randn=False):

        self.link2labels = {k: self._choose_labels(randn=randn) for k in self.links}

        return self

    def populate_node2links(self):

        self.node2links = {}
        for n in self.nodes:
            lab = np.random.choice(self.labels)
            self.node2links[n] = self._choose_links(self.label2links[lab])
            self.node2labels[n] = {lab}

        return self

    def map_link2nodes(self):

        self.link2nodes = {}
        for n, links in self.node2links.items():
            for l in links:
                if l in self.link2nodes:
                    self.link2nodes[l] |= {n}
                else:
                    self.link2nodes[l] = {n}

        return self

    def map_label2links(self):

        self.label2links = {}
        for l, labs in self.link2labels.items():
            for lab in labs:
                if lab in self.label2links:
                    self.label2links[lab].append(l)
                else:
                    self.label2links[lab] = [l]

        for lab, links in self.label2links.items():
            self.label2links[lab] = list(set(links))

        return self

    def _choose_labels(self, randn=False):

        return set(np.random.choice(self.labels, np.random.randint(1,self.n_labels) if randn else 1, replace=False))

    def _choose_links(self, links=None, n=None):

        if not links:
            links = self.links

        if n is None:
            n = np.random.randint(self.linkrange[0], self.linkrange[1])

        return set(np.random.choice(links, n, replace=False))

    def _choose_nodes(self, nodes=None, n=1):

        if not nodes:
            nodes = self.nodes

        return set(np.random.choice(nodes, n, replace=False))

    def multilabel_nodes(self, ratio=0.2):

        links = self._choose_links(n=int(np.ceil(self.n_links * ratio)))

        for l in links:
            self.link2labels[l] |= self._choose_labels(randn=True)
            if l in self.link2nodes:
                for n in self.link2nodes[l]:
                    self.node2labels[n] |= self.link2labels[l]

        return self

    def write(self, p_data='', labels=True, writemode='w+', header='', dir_dest=''):

        if not p_data:
            dataid = ''.join(np.random.choice(list('abcdefgh12345678'), 6))
            dir_dest += '/' if not dir_dest.endswith('/') else ''
            if header:
                p_data = dir_dest + 'dummy_%s-%s.tsv' % (header, dataid)
            else:
                p_data = dir_dest + 'dummy_network-%s.tsv' % dataid

        with open(p_data, writemode) as f:

            if labels:
                if 'a' not in writemode:
                    _ = f.write('nodes\tlinks\tlabels\n')
                for n in self.nodes:
                    lab = self.node2labels[n]
                    links = self.node2links[n]
                    _ = f.write('%s\t%s\t%s\n' % (n, '/'.join(links), '/'.join(lab)))

            else:
                if 'a' not in writemode:
                    _ = f.write('nodes\tlinks\n')
                for n in self.nodes:
                    links = self.node2links[n]
                    _ = f.write('%s\t%s\n' % (n, '/'.join(links)))

        return p_data
