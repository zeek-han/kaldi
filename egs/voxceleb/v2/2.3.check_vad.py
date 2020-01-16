#!/usr/bin/env python3

import os
import sys
from glob import iglob, glob
from os.path import basename, dirname, join as p_join
import json

import numpy as np
from kaldiio import ReadHelper, WriteHelper



vad_scp = 'mfcc/vad_train.1.scp'
mfcc_scp = 'mfcc/raw_mfcc_train.1.scp'

utt_ids = []
vads = []
utt_id2vad = {}
utt_id2mfcc = {}

#low_resol_vad = np.load('/media/sangjik/hdd2/dataset/speech/English/VoxCeleb2/npy_vad/vad_npy/train.17/id04462-35ZDCA1bA0U-00001.npy')
#low_resol_vad = np.load('/media/sangjik/hdd2/dataset/speech/English/VoxCeleb2/npy_vad/vad_npy/train.1/id00026-KPiwotirhuQ-00017.npy')
#print(low_resol_vad)
#print(np.where(low_resol_vad == 0))
#print(len(np.where(low_resol_vad == 0)[0]))

with ReadHelper('scp:' + vad_scp) as vad_reader:
    for i, (utt_id, vad) in enumerate(vad_reader):
        #if i < 1:
        if i < 1e7:
        #if utt_id == 'id00026-KPiwotirhuQ-00017':
            utt_id2vad[utt_id] = vad
            #print(utt_id, vad.shape)
            #print(type(utt_id), utt_id=='id00026-KPiwotirhuQ-00017', '{}:{}'.format(utt_id, np.where(vad ==0)))
            #print(utt_id, vad.shape)
        else:
            break

with ReadHelper('scp:' + mfcc_scp) as mfcc_reader:
    for utt_id, mfcc in mfcc_reader:
        #if i < 1:
        if utt_id == 'id00026-KPiwotirhuQ-00017':
            utt_id2mfcc[utt_id] = mfcc
            break

utt_id='id00026-KPiwotirhuQ-00017'

print(utt_id2vad[utt_id].shape)
with WriteHelper('ark,scp:{},{}'.format('./tmp/vad.ark', './tmp/vad.scp')) as writer:
    writer[utt_id] = utt_id2vad[utt_id]

print(utt_id2mfcc[utt_id].shape)
with WriteHelper('ark,scp:{},{}'.format('./tmp/mfcc.ark', './tmp/mfcc.scp')) as writer:
    writer[utt_id] = utt_id2mfcc[utt_id]


#print('vad reading completed')
#with ReadHelper('scp:' + mfcc_scp) as mfcc_reader:
#    for i, (utt_id, mfcc) in enumerate(mfcc_reader):
#        if utt_id in utt_id2vad.keys():
#            vad = utt_id2vad[utt_id]
#            assert(len(vad) == len(mfcc))
#            print(i, 'check',  vad.shape, mfcc.shape)
