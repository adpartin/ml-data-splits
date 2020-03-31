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
from datasplit.cv_splitter import cv_splitter
from utils.plots import plot_hist
from utils.utils import load_data, dump_dict, get_print_func


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate and save train/val/test data splits.')

    # Input data
    parser.add_argument('-dp', '--datapath', required=True, default=None, type=str,
                        help='Full path to the data (default: None).')

    # Out split path
    parser.add_argument('--gout', default=None, type=str,
                        help='Global outdir. Dir to dump the splits.')

    # Data split methods
    # parser.add_argument('-tem', '--te_method', default='simple', choices=['simple', 'group', 'strat'],
    #                     help='Test split method (default: simple).')
    parser.add_argument('-cvm', '--cv_method', default='simple', choices=['simple', 'group', 'strat'],
                        help='Cross-val split method (default: simple).')
    parser.add_argument('--te_size', type=float, default=0.1, help='Test size split ratio (default: 0.1).')
    # parser.add_argument('--vl_size', type=float, default=0.1, help='Val size split ratio for single split (default: 0.1).')

    parser.add_argument('--split_on', type=str, default=None, choices=['cell', 'drug'],
                        help='Specify how to make a hard split (default: None).')
    parser.add_argument('-ns', '--n_splits', type=int, default=5, help='Number of splits to generate (default: 5).')

    parser.add_argument('--ml_task', type=str, default='reg', choices=['reg', 'cls'], help='ML task (default: reg).')
    parser.add_argument('-t', '--trg_name', type=str, default=None,
                        help='Target column name (required when stratify) (default: None).')

    # Other
    parser.add_argument('--n_jobs', default=8,  type=int, help='Default: 8.')

    # Parse args and run
    args, other_args = parser.parse_known_args(args)
    return args


def split_size(x):
    """ Split size can be float (0, 1) or int (casts value as needed). """
    assert x > 0, 'Split size must be greater than 0.'
    return int(x) if x > 1.0 else x


def print_intersect_on_var(df, tr_id, vl_id, te_id, grp_col='CELL', print_fn=print):
    """ Print intersection between train, val, and test datasets with respect
    to grp_col column if provided. df is usually metadata.
    """
    if grp_col in df.columns:
        tr_grp_unq = set(df.loc[tr_id, grp_col])
        vl_grp_unq = set(df.loc[vl_id, grp_col])
        te_grp_unq = set(df.loc[te_id, grp_col])
        print_fn(f'\tTotal intersects on {grp_col} btw tr and vl: {len(tr_grp_unq.intersection(vl_grp_unq))}')
        print_fn(f'\tTotal intersects on {grp_col} btw tr and te: {len(tr_grp_unq.intersection(te_grp_unq))}')
        print_fn(f'\tTotal intersects on {grp_col} btw vl and te: {len(vl_grp_unq.intersection(te_grp_unq))}')
        print_fn(f'\tUnique {grp_col} in tr: {len(tr_grp_unq)}')
        print_fn(f'\tUnique {grp_col} in vl: {len(vl_grp_unq)}')
        print_fn(f'\tUnique {grp_col} in te: {len(te_grp_unq)}')    
    else:
        raise(f'The column {grp_col} was not found!')

    
def data_splitter(data, cv_method='simple', te_method='simple',
                  n_splits=10, te_size=0.1, mltype='reg', gout=Path('./'),
                  outfigs=Path('./'), split_on=None, ydata=None, trg_name=None,
                  print_fn=print):
    # TODO: put the spltting subroutine into function
    for seed in range( n_splits ):
        # digits = len(str(n_splits))
        seed_str = str(seed) # f"{seed}".zfill(digits)
        output = '1fold_s' + seed_str 

        # Note that we don't shuffle the original dataset, but rather
        # create a vector array of representative indices.
        np.random.seed( seed )
        idx_vec = np.random.permutation( data.shape[0] )
        y_vec = ydata.values[idx_vec]
        
        # Create splitter that splits the full dataset into tr and te
        te_folds = int(1/te_size)
        te_splitter = cv_splitter(cv_method=te_method, cv_folds=te_folds, test_size=None,
                                  mltype=mltype, shuffle=False, random_state=seed)
        
        te_grp = None if split_on is None else data[split_on].values[idx_vec]
        if is_string_dtype(te_grp): te_grp = LabelEncoder().fit_transform(te_grp)
        
        # Split tr into tr and te
        tr_id, te_id = next(te_splitter.split(X=idx_vec, y=y_vec, groups=te_grp))
        tr_id = idx_vec[tr_id] # adjust the indices! we'll split the remaining tr into tr and vl
        te_id = idx_vec[te_id] # adjust the indices!

        # Update a vector array that excludes the test indices
        idx_vec_ = tr_id; del tr_id
        y_vec_ = ydata.values[idx_vec_]

        # Define vl_size while considering the new full size of the available samples
        vl_size = te_size / (1 - te_size)
        cv_folds = int(1/vl_size)

        # Create splitter that splits tr into tr and vl
        cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=None,
                         mltype=mltype, shuffle=False, random_state=seed)    
        
        cv_grp = None if split_on is None else data[split_on].values[idx_vec_]
        if is_string_dtype(cv_grp): cv_grp = LabelEncoder().fit_transform(cv_grp)
        
        # Split tr into tr and vl
        tr_id, vl_id = next(cv.split(X=idx_vec_, y=y_vec_, groups=cv_grp))
        tr_id = idx_vec_[tr_id] # adjust the indices!
        vl_id = idx_vec_[vl_id] # adjust the indices!
        
        # Dump tr, vl, te indices
        np.savetxt(gout/f'{output}_tr_id.csv', tr_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n')
        np.savetxt(gout/f'{output}_vl_id.csv', vl_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n')
        np.savetxt(gout/f'{output}_te_id.csv', te_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n')
        
        # Check that indices do not overlap
        assert len( set(tr_id).intersection(set(vl_id)) ) == 0, 'Overlapping indices btw tr and vl'
        assert len( set(tr_id).intersection(set(te_id)) ) == 0, 'Overlapping indices btw tr and te'
        assert len( set(vl_id).intersection(set(te_id)) ) == 0, 'Overlapping indices btw tr and vl'
        
        print_fn('Train samples {} ({:.2f}%)'.format( len(tr_id), 100*len(tr_id)/data.shape[0] ))
        print_fn('Val   samples {} ({:.2f}%)'.format( len(vl_id), 100*len(vl_id)/data.shape[0] ))
        print_fn('Test  samples {} ({:.2f}%)'.format( len(te_id), 100*len(te_id)/data.shape[0] ))
        
        # Confirm that group splits are correct (no intersect)
        if split_on is not None:
            print_intersect_on_var(data, tr_id=tr_id, vl_id=vl_id, te_id=te_id, grp_col=split_on, print_fn=print_fn)

        if trg_name in data.columns:
            plot_hist(data.loc[tr_id, trg_name], title=f'Train Set; Histogram; {trg_name}',
                      fit=None, bins=100, path=outfigs/f'{output}_AUC_hist_train.png')
            plot_hist(data.loc[vl_id, trg_name], title=f'Val Set; Histogram; {trg_name}',
                      fit=None, bins=100, path=outfigs/f'{output}_AUC_hist_val.png')
            plot_hist(data.loc[te_id, trg_name], title=f'Test Set; Histogram; {trg_name}',
                      fit=None, bins=100, path=outfigs/f'{output}_AUC_hist_test.png')
    return None


def run(args):
    t0 = time()
    te_size = split_size( args['te_size'] )
    n_splits = int( args['n_splits'] )
    # args['datapath'] = str(Path(args['datapath']).absolute())
    args['datapath'] = Path( args['datapath'] ).resolve()
    datapath = args['datapath']

    # Hard split
    split_on = None if args['split_on'] is None else args['split_on'].upper()
    cv_method = args['cv_method']
    te_method = cv_method 

    # Specify ML task (regression or classification)
    if cv_method=='strat':
        mltype = 'cls'  # force mltype to cls in case of stratification
    else:
        mltype = args['ml_task']  

    # Target column name
    trg_name = str( args['trg_name'] )
    
    # -----------------------------------------------
    #       Create outdir
    # -----------------------------------------------
    # datapath = Path(args['datapath']).absolute().parent
    if args['gout'] is not None:
        gout = Path( args['gout'] ).resolve()
    else:
        # split_str_on_sep = str( datapath ).split('/data/')
        # dir1 = split_str_on_sep[0] + '/trn'
        # dir2 = Path( split_str_on_sep[1] ).with_suffix('')
        # gout = Path(dir1, dir2, 'splits')
        # TODO: useful for drug response
        # sufx = 'none' if split_on is None else split_on
        # gout = gout / f'split_on_{sufx}'
        gout = Path( str( datapath.with_suffix('') ) + '.splits' )
    
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
    dump_dict(args, outpath=gout/'data.splitter.args.txt') # dump args.
    
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

    data_splitter(data=data, cv_method='simple', te_method='simple',
                  n_splits=n_splits, te_size=te_size, mltype=mltype,
                  gout=gout, outfigs=outfigs, split_on=split_on,
                  ydata=ydata, trg_name=trg_name, print_fn=print_fn)

    print_fn('Runtime: {:.1f} min'.format( (time()-t0)/60) )
    print_fn('Done.')
    lg.kill_logger()
    
    
def main(args):
    args = parse_args(args)
    args = vars(args)
    ret = run(args)
    
    
if __name__ == '__main__':
    main(sys.argv[1:])


