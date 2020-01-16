#!/usr/bin/env python3

import os
from os.path import basename, dirname, join as p_join
from glob import glob
import json

import numpy as np
from kaldiio import ReadHelper, WriteHelper

def average_pool(mfcc, window=10):
    len_pooled_mfcc = int(len(mfcc) / window)
    pooled_mfcc = np.zeros((len_pooled_mfcc, mfcc.shape[1]), dtype=mfcc.dtype)
    for i in range(len_pooled_mfcc):
        pooled_mfcc[i] = np.mean(mfcc[(i * window):((i + 1) * window)], axis=0)
    return pooled_mfcc

def max_pool(mfcc, window=10):
    len_pooled_mfcc = int(len(mfcc) / window)
    pooled_mfcc = np.zeros((len_pooled_mfcc, mfcc.shape[1]), dtype=mfcc.dtype)
    for i in range(len_pooled_mfcc):
        pooled_mfcc[i] = np.max(mfcc[(i * window):((i + 1) * window)], axis=0)
    return pooled_mfcc

def rewrite(name, mfcc_dir, mfcc_scp_dir, var_pool):
    os.makedirs(mfcc_dir, exist_ok=True)
    with open(name + '.json', 'r') as fp:
        utt_id2mfcc = json.load(fp)
#####################
    for scp in glob(p_join(mfcc_scp_dir, 'raw_mfcc_{}.*.scp'.format(name))):
    #for i in range(1):
    #    scp = './mfcc_scp_total_default.amend_path/raw_mfcc_voxceleb1_test.1.scp'
#####################
        with open(scp, 'r') as scp_f:
            utt_ids = sorted([line.split(' ')[0] for line \
                                in scp_f.read().split('\n') if len(line) > 0])
        new_scp = p_join(mfcc_dir, basename(scp))
        new_ark = new_scp[:-4] + '.ark'
        #print(utt_ids[0], len(utt_ids), new_scp, new_ark)
        with WriteHelper('ark,scp:{},{}'.format(new_ark, new_scp)) as writer:
            for i in range(len(utt_ids)):
                utt_id = utt_ids[i]
                try:
                    mfcc_npy = utt_id2mfcc[utt_id]
                    if var_pool == 'average':
                        pooled_mfcc = average_pool(np.load(mfcc_npy))
                    elif var_pool == 'max':
                        pooled_mfcc = max_pool(np.load(mfcc_npy))
                    else:
                        print('different_var_pool:', var_pool)
                    writer[utt_id] = pooled_mfcc
                    print(scp, '--', var_pool, "_pooled_mfcc:", pooled_mfcc.shape)
                except KeyError:
                    #writer[utt_id] = np.load(p_join('./tmp_wav', utt_id + '.npy'))
                    print('KeyError:', utt_id)

if __name__ == '__main__':
    #from_mfcc_dir = 'kaldi/egs/voxceleb/v2.smallest/mfcc/'
    #mfcc_npy_root_dir = '/home/npy_mfcc'
    #######mfcc_before_pool_dir = '/home/npy_mfcc'
    #######mfcc_after_pool_dir = '/home/sangjik/npy_after_pool_djt_mfcc'

    mfcc_output_dir = '/home/sangjik/kaldi/egs/voxceleb/v2/mfcc.average'
    mfcc_scp_dir    = '/home/sangjik/kaldi/egs/voxceleb/v2/mfcc_scp'

    rewrite('train',          mfcc_output_dir, mfcc_scp_dir, 'average')
    rewrite('voxceleb1_test', mfcc_output_dir, mfcc_scp_dir, 'average')
    rewrite('train_aug_1m',   mfcc_output_dir, mfcc_scp_dir, 'average')

    #rewrite('train',          mfcc_output_dir, mfcc_scp_dir, 'max')
    #rewrite('voxceleb1_test', mfcc_output_dir, mfcc_scp_dir, 'max')
    #rewrite('train_aug_1m',   mfcc_output_dir, mfcc_scp_dir, 'max')
