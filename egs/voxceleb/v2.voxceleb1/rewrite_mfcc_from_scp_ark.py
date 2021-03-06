#!/usr/bin/env python3

import os, sys
from os.path import basename, dirname, join as p_join
from glob import glob
import json

import numpy as np
from kaldiio import ReadHelper, WriteHelper


def extract_mfcc(name, original_mfcc_dir, mfcc_npy_root_dir):
    utt_id2mfcc = {}
    for scp in glob(p_join(original_mfcc_dir, 'raw_mfcc_{}.*.scp'.format(name))):
        num = scp.split('.')[-2]
        print('extract:', scp)
        mfcc_npy_dir = p_join(mfcc_npy_root_dir, name + '.' + num)
        os.makedirs(mfcc_npy_dir, exist_ok=True)
        with ReadHelper('scp:' + scp) as reader:
            for utt_id, mfcc in reader:
                #print(utt_id, mfcc.shape)
                mfcc_npy = p_join(mfcc_npy_dir, utt_id + '.npy')
                #print(mfcc_npy)
                np.save(mfcc_npy, mfcc)
                utt_id2mfcc[utt_id] = mfcc_npy
    return utt_id2mfcc

def rewrite(name, mfcc_dir, mfcc_scp_dir):
    os.makedirs(mfcc_dir, exist_ok=True)
    with open(name + '.json', 'r') as fp:
        utt_id2mfcc = json.load(fp)
    for scp in glob(p_join(mfcc_scp_dir, 'raw_mfcc_{}.*.scp'.format(name))):
    #for scp in glob(p_join('./mfcc_scp', 'raw_mfcc_{}.*.scp'.format(name))):
        with open(scp, 'r') as scp_f:
            utt_ids = sorted([line.split(' ')[0] for line \
                                in scp_f.read().split('\n') if len(line) > 0])
        #print('rewrite:', scp, len(utt_ids), utt_ids[0], utt_id2mfcc[utt_ids[0]])
        new_scp = p_join(mfcc_dir, basename(scp))
        new_ark = new_scp[:-4] + '.ark'

        with WriteHelper('ark,scp:{},{}'.format(new_ark, new_scp)) as writer:
            for i in range(len(utt_ids)):
                utt_id = utt_ids[i]
                try:
                    mfcc_npy = utt_id2mfcc[utt_id]
                    writer[utt_id] = np.load(mfcc_npy)
                except KeyError:
                    writer[utt_id] = np.load(p_join('./tmp_wav', utt_id + '.npy'))
                    print('KeyError:', utt_id)

if __name__ == '__main__':
    from_mfcc_dir = sys.argv[1]
    mfcc_npy_root_dir = sys.argv[2]
    mfcc_output_dir = sys.argv[3]
    mfcc_scp_dir = sys.argv[4]

    utt_id2mfcc = extract_mfcc('train', from_mfcc_dir, mfcc_npy_root_dir)
    with open('train.json', 'w') as fp:
        json.dump(utt_id2mfcc, fp)
    rewrite('train', mfcc_output_dir, mfcc_scp_dir)

    utt_id2mfcc = extract_mfcc('voxceleb1_test', from_mfcc_dir, mfcc_npy_root_dir)
    with open('voxceleb1_test.json', 'w') as fp:
        json.dump(utt_id2mfcc, fp)
    rewrite('voxceleb1_test', mfcc_output_dir, mfcc_scp_dir)
