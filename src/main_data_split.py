""" 
This code generates multiple splits of train/val/test sets.
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from time import time
from pprint import pprint, pformat

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder

# File path
filepath = Path(__file__).resolve().parent

# Utils
from utils.classlogger import Logger
from datasplit.splitter import data_splitter # cv_splitter
from utils.plots import plot_hist
from utils.utils import load_data, dump_dict, get_print_func


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate and save train/val/test data splits.')

    parser.add_argument('-dp', '--datapath', required=True, default=None, type=str,
                        help='Full path to the data (default: None).')
    parser.add_argument('--gout', default=None, type=str,
                        help='Global outdir to dump the splits.')
    
    parser.add_argument('-ns', '--n_splits', type=int, default=5, help='Number of splits to generate (default: 5).')

    # parser.add_argument('-tem', '--te_method', default='simple', choices=['simple', 'group', 'strat'],
    #                     help='Test split method (default: simple).')
    parser.add_argument('-cvm', '--cv_method', default='simple', choices=['simple', 'group', 'strat'],
                        help='Cross-val split method (default: simple).')
    parser.add_argument('--te_size', type=float, default=0.1, help='Test size split ratio (default: 0.1).')
    # parser.add_argument('--vl_size', type=float, default=0.1, help='Val size split ratio for single split (default: 0.1).')

    parser.add_argument('--split_on', type=str, default=None, choices=['cell', 'drug'],
                        help='Specify which variable (column) to make a hard split on (default: None).')

    parser.add_argument('--ml_task', type=str, default='reg', choices=['reg', 'cls'], help='ML task (default: reg).')
    parser.add_argument('-t', '--trg_name', type=str, default=None,
                        help='Target column name (required when stratify) (default: None).')
 
    parser.add_argument('--n_jobs', default=8,  type=int, help='Default: 8.')
    args, other_args = parser.parse_known_args(args)
    return args


def run(args):
    t0 = time()
    n_splits = int( args['n_splits'] )
    ## te_size = split_size( args['te_size'] )
    te_size = args['te_size']
    args['datapath'] = Path( args['datapath'] ).resolve()
    datapath = args['datapath']

    # Hard split
    split_on = None if args['split_on'] is None else args['split_on'].upper()
    cv_method = args['cv_method']
    te_method = cv_method 

    # Specify ML task (regression or classification)
    if cv_method=='strat':
        mltype = 'cls'  # cast mltype to cls in case of stratification
    else:
        mltype = args['ml_task']  

    # Target column name
    trg_name = str( args['trg_name'] )
    
    # -----------------------------------------------
    #       Create outdir
    # -----------------------------------------------
    if args['gout'] is not None:
        gout = Path( args['gout'] ).resolve()
        gout = gout / datapath.with_suffix('.splits').name
    else:
        # Note! useful for drug response
        # sufx = 'none' if split_on is None else split_on
        # gout = gout / f'split_on_{sufx}'
        gout = datapath.with_suffix('.splits')
    
    outfigs = gout / 'outfigs'
    os.makedirs(gout, exist_ok=True)
    os.makedirs(outfigs, exist_ok=True)

    # -----------------------------------------------
    #       Create logger
    # -----------------------------------------------
    lg = Logger(gout/'data.splitter.log')
    print_fn = get_print_func(lg.logger)
    print_fn(f'File path: {filepath}')
    print_fn(f'\n{pformat(args)}')
    dump_dict(args, outpath=gout/'data.splitter.args.txt') # dump args
    
    # -----------------------------------------------
    #       Load data
    # -----------------------------------------------
    print_fn('\nLoad master dataset.')
    data = load_data( datapath )
    print_fn('data.shape {}'.format(data.shape))
    print_fn('Total mod: {}'.format( len([c for c in data.columns if 'mod.' in c]) ))
    
    ydata = data[trg_name] if trg_name in data.columns else None
    if ydata is None and cv_method=='strat':
        raise ValueError('Y data must be available if splits are required to stratified.')
    if ydata is not None:
        plot_hist(ydata, title=f'{trg_name}', fit=None, bins=100, path=outfigs/f'{trg_name}_hist_all.png')
     
    # -----------------------------------------------
    #       Generate splits (train/val/test)
    # -----------------------------------------------
    print_fn('\n{}'.format('-'*50))
    print_fn('Split into hold-out train/val/test')
    print_fn('{}'.format('-'*50))

    kwargs = {'data': data, 'cv_method': cv_method, 'te_method': te_method,
              'te_size': te_size, 'mltype': mltype, 'split_on': split_on}

    data_splitter(n_splits=n_splits, gout=gout, outfigs=outfigs, ydata=ydata,
                  print_fn=print_fn, **kwargs)

    print_fn('Runtime: {:.1f} min'.format( (time()-t0)/60) )
    print_fn('Done.')
    lg.kill_logger()
    
    
def main(args):
    args = parse_args(args)
    args = vars(args)
    ret = run(args)
    
    
if __name__ == '__main__':
    main(sys.argv[1:])


