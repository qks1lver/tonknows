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
        for k in self.nodes:
            lab = np.random.choice(self.labels)
            self.node2links[k] = self._choose_links(self.label2links[lab])
            self.node2labels[k] = [lab]

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

    def _choose_links(self, links=None):

        if not links:
            links = self.links

        return set(np.random.choice(links, np.random.randint(self.linkrange[0], self.linkrange[1]), replace=False))

    def write(self, p_data=''):

        if not p_data:
            p_data = 'dummy_network-%s.tsv' % ''.join(np.random.choice(list('abcdefgh12345678'), 6))

        with open(p_data, 'w+') as f:
            _ = f.write('nodes\tlinks\tlabels\n')
            for n in self.nodes:
                lab = self.node2labels[n]
                links = self.node2links[n]
                _ = f.write('%s\t%s\t%s\n' % (n, '/'.join(links), '/'.join(lab)))

        return p_data
