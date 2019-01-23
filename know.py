#!/usr/bin/env python3

"""
know.py

This is the main script to run.

TonKnows is a light-weight, fast, and simple tool to predict the missing information in your network.

MIT License. Copyright 2018 Jiun Y. Yen (jiunyyen@gmail.com)
"""


# Import
import os
import argparse
from src.model import Model
from src.iofunc import *
from src.test import Dummy
import pdb


# Set default directory and current model paths \_______________________________________________________________________
# tonknows directory
_d_tonknows_ = os.path.realpath(os.path.split(__file__)[0]) + '/'

# tonknows models directory
_d_model_ = _d_tonknows_ + 'models/'
if not os.path.isdir(_d_model_):
    os.makedirs(_d_model_)

# current model
_p_current_model_ = ''
for p in os.listdir(_d_model_):
    if 'current' in p:
        _p_current_model_ = _d_model_ + p
        break

# tmp directory
_d_tmp_ = _d_tonknows_ + 'tmp/'
if not os.path.isdir(_d_tmp_):
    os.makedirs(_d_tmp_)

# test directory
_d_test_ = _d_tonknows_ + 'test/'
if not os.path.isdir(_d_test_):
    os.makedirs(_d_test_)


# Main \________________________________________________________________________________________________________________
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tonknows')
    parser.add_argument('-v', dest='to_verbose', action='store_true', help='To verbose')
    parser.add_argument('--mod', dest='model', default=_p_current_model_, help='Model file')
    parser.add_argument('--td', dest='train_data', default='', help= 'Train data file')
    parser.add_argument('--ed', dest='eval_data', default='', help='Evaluation data file')
    parser.add_argument('--pd', dest='pred_data', default='', help='Data to predict for')
    parser.add_argument('--out', dest='model_out', default='', help='Output model filename override')
    parser.add_argument('--cvlayers', dest='cv_layers', action='store_true', help='CV over each layer')
    parser.add_argument('--multilayer', dest='multilayer', action='store_true', help='Train multilayer')
    parser.add_argument('--onlylabels', dest='only_labs', default='', help='Labels to use exclusively, comma separated')
    parser.add_argument('--notlabels', dest='not_labs', default='', help='Labels to exclude, comma separated')
    parser.add_argument('--param', dest='param', default='', help='Use param file')
    parser.add_argument('--kcv', dest='kcv', default=10, type=int, help='Number of CV k-fold, minimum 3')
    parser.add_argument('--plot', dest='to_plot', action='store_true', help='To plot when it can plot')
    parser.add_argument('--nrep', dest='n_repeat', default=3, type=int, help='N times to repeat train/eval')
    parser.add_argument('--maxlinkratio', dest='maxlinkratio', default=0.25, type=float, help='Maximum link ratio to use as link')
    parser.add_argument('--minlinkfreq', dest='minlinkfreq', default=1, type=int, help='Minimum link frequency to use as link')
    parser.add_argument('--aim', dest='aim', default=None, help='To specifiy aim of the model')
    parser.add_argument('--save', dest='save', action='store_true', help='Save data')
    parser.add_argument('--open', dest='open', default='', help='Open saved pickle data')
    parser.add_argument('--shell', dest='shell', action='store_true', help='Python shell interactive session')
    parser.add_argument('--analyze', dest='analyze', default='', help='Analyze data, broad options')
    parser.add_argument('--data', dest='data', default='', help='Folder or path of data to analyze')
    parser.add_argument('--setcurrent', dest='setcurrent', action='store_true', help='Set model to current after training')
    parser.add_argument('--usef1', dest='use_f1', action='store_true', help='Use F1-score when plotting')
    parser.add_argument('--test', dest='test', action='store_true', help='For testing anything')

    args = parser.parse_args()

    # header
    print('\n\n____/   T o n K n o w s   \______________________________________________________________________')
    print('    \  the network miner  /\n\n')

    # Initialize variable description in case going into interactive mode
    vd = {}

    # Import param
    param = loadparam(args.param, _d_tmp_)

    # Train --------------------------------------------------------------------------------------------------------
    if args.train_data:

        # Use full path
        args.train_data = os.path.realpath(args.train_data) if args.train_data else ''

        # Subcellular compartments of interest
        if args.only_labs:
            labels = [x.strip() for x in args.only_labs.split(',')]
        elif args.not_labs:
            m = Model(data=args.train_data, verbose=False, columns=param['columns'] if args.param and 'columns' in param else None).load_data()
            labels = list(set(m.datas[m.train_idx].locations) - set(args.not_labs.strip().split(',')))
        else:
            labels = None

        # Initialize model
        m = Model(data=param['datamod'](args.train_data) if args.param and 'datamod' in param else args.train_data,
                  exclude_links=param['exclude_links'] if args.param and 'exclude_links' in param else None,
                  labels=param['labels'] if args.param and 'labels' in param else labels,
                  verbose=args.to_verbose,
                  columns=param['columns'] if args.param and 'columns' in param else None,
                  aim=args.aim)
        m.maxlidxratio = args.maxlinkratio
        m.minlinkfreq = args.minlinkfreq
        m.masklayer = param['masklayer'] if args.param and 'masklayer' in param else []
        m.load_data()

        if args.not_labs:
            # make sure excluded labels don't go into the "other" label
            if len(args.not_labs.split(',')) == 1:
                m.datas[m.train_idx].lab_other = False

        # To CV each layer must be training multilayer data
        if args.cv_layers:
            args.multilayer = True

        # Whether to train with training dataset as multilayer network
        m.train_multilayers = args.multilayer

        # Setup parameters
        if args.kcv < 3:
            print('\n { Need at least 3-fold CV: 1 train base classifiers, 1 train final classifer, 1 evaluate. Increased to kcv=3 }\n')
            args.kcv = 3

        m.kfold_cv = args.kcv
        m.n_repeat = args.n_repeat

        # Run train
        if args.cv_layers:
            res = m.train_cv_layers()
        else:
            res = m.train()

        # Save model and training results
        if args.save:
            print('\n__  Saving model \_______________________________')
            print('  \ Model ID: %s' % m.id)

            current_tag = ''
            if args.setcurrent:
                if _p_current_model_:
                    print('    ** Replacing current model: %s' % _p_current_model_)
                    os.rename(_p_current_model_, _p_current_model_.replace('-current',''))
                current_tag = '-current'

            if args.model_out:
                filename = args.model_out
            else:
                filename = m.id

            p_mod = '%s%s%s.pkl' % (_d_model_, filename, current_tag)
            if args.setcurrent:
                _p_current_model_ = p_mod

            m_pkg = m.export_model()
            save_pkl(p_mod, [m_pkg, res])
            print('   \ Saved: %s' % p_mod)

        # set vd and local variables
        vd = {'m': 'trained model (Model class)',
              'res': 'training results (Pandas dataframe)'}

    # Eval on separate dataset  ------------------------------------------------------------------------------------
    if args.eval_data:

        args.eval_data = param['datamod'](args.eval_data) if args.param and 'datamod' in param else os.path.realpath(args.eval_data)
        args.model = os.path.realpath(args.model) if args.model else _p_current_model_

        if not args.model:
            print('\n { Missing model file, use --mod path/to/model.pkl }\n')
        else:
            print('\n__  Loading model \_______________________________\n  \ File: %s \n' % args.model)
            m_pkg, _ = open_pkl(args.model)
            m = Model(verbose=args.to_verbose)
            m.import_model(m_pkg)
            m.add_data(data=args.eval_data, build=False)
            m.datas[-1].mimic(data=m.datas[m.train_idx])
            m.datas[-1].build_data()

            print('\n__  Building data \_______________________________\n  \ File: %s \n' % args.eval_data)
            m.train_multilayers = False
            res = m.eval(data=m.datas[-1])

            # set vd
            vd = {'m': 'loaded model (Model class)',
                  'res': 'evaluation results (dictionary)'}

    # Predict ------------------------------------------------------------------------------------------------------
    if args.pred_data:

        args.pred_data = param['datamod'](args.pred_data) if args.param and 'datamod' in param else os.path.realpath(args.pred_data)
        args.model = os.path.realpath(args.model) if args.model else _p_current_model_

        if not args.model:
            print('\n { Missing model file, use --mod path/to/model.pkl }\n')
        else:
            print('\n__  Loading model \_______________________________\n  \ File: %s \n' % args.model)
            m_pkg, _ = open_pkl(args.model)
            m = Model(verbose=args.to_verbose)
            m.import_model(m_pkg)
            m.add_data(data=args.pred_data, build=False)
            m.datas[-1].mimic(data=m.datas[m.train_idx])
            m.datas[-1].build_data()

            print('\n__  Building data \_______________________________\n  \ File: %s \n' % args.pred_data)
            m.train_multilayers = False
            res, p_out = m.predict(data=m.datas[-1])

            # set vd
            vd = {'m': 'loaded model (Model class)',
                  'res': 'prediction results (dictionary)',
                  'p_out': 'prediction results saved path'}

    # Open --------------------------------------------------------------------------------------------------------
    if args.open:
        m_pkg, res = open_pkl(args.open)
        m = Model()
        m.import_model(m_pkg)
        # set vd
        vd = {'m':'loaded model (Model class)',
              'res':'loaded training results (Pandas dataframe)'}

    # Analyze ------------------------------------------------------------------------------------------------------
    if args.analyze:
        from src.analysis import Analysis

        if args.analyze == 'model':
            # This analyze the training results stored with each model in the .pkl
            if args.data:
                analysis = Analysis(p_data=args.data)
            else:
                analysis = Analysis(p_data=_d_model_)
            analysis.use_f1 = args.use_f1
            data = analysis.compile_batch_train_results()

            vd = {'data': 'Compiled training data (Pandas dataframe)'}


    # Enable interaction \______________________________________________________________________________________________
    if args.shell:
        interact(var_desc=vd, local=locals())


    # For testing new code \____________________________________________________________________________________________
    if args.test:
        d = Dummy().init_networks()
        d.write(_d_test_ + 'pure_network.tsv')
        d.multilabel_nodes(ratio=0.2).write(_d_test_ + 'mix_network.tsv')
