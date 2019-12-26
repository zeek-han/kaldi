#!/usr/bin/env python3

from os.path import basename, dirname, join as p_join
import json
from datetime import datetime

import numpy as np
from kaldiio import ReadHelper, WriteHelper

num_job = 12
prefix = '/media/sangjik/hdd2/dataset/speech/English/VoxCeleb2'
mfcc_output_dir = '/media/sangjik/hdd1/zeek_github/ASR/kaldi/egs/voxceleb/v2/mfcc'

#for name in ['voxceleb1_test']:
for name in ['train']:
#for name in ['train_aug_1m']:
    print('           - name:', name)
    id2mfcc_json = '{}/{}.json'.format(prefix, name)
    with open(id2mfcc_json, 'r') as fd:
        utt_id2mfcc = json.load(fd)
    print('           - utt_id2mfcc json loaded')

    for ii in range(num_job):
        scp="/media/sangjik/hdd1/zeek_github/ASR/kaldi/egs/voxceleb/v2/mfcc_scp/raw_mfcc_{}.{}.scp".format(name, ii + 1)
        utt_ids = []
        with open(scp, 'r') as fd:
            for line in fd:
                utt_id = line[:-1].split()[0]
                utt_ids.append(utt_id)
                #print(utt_id, utt_id2mfcc[utt_id])
        
        new_scp = p_join(mfcc_output_dir, basename(scp))
        now = datetime.now()
        print('{}-{}-{}T{}:{}:{}, ii={}'.format(now.hour, now.month, now.day, 
                                   now.hour, now.minute, now.second, ii),  new_scp)
        new_ark = new_scp[:-4] + '.ark'

        with WriteHelper('ark,scp:{},{}'.format(new_ark, new_scp)) as writer:
            for i in range(len(utt_ids)):
                utt_id = utt_ids[i]
                try:
                    mfcc_npy = p_join(prefix, utt_id2mfcc[utt_id])
                    writer[utt_id] = np.load(mfcc_npy)
                except KeyError:
                    #writer[utt_id] = np.load(p_join('./tmp_wav', utt_id + '.npy'))
                    print('KeyError:', utt_id)

