#!/usr/bin/python3

import os
import sys
from glob import iglob, glob
from os.path import basename, dirname, join as p_join
import json

import numpy as np
from kaldiio import ReadHelper, WriteHelper



#vad_scp = 'mfcc/vad_train.1.scp'
#mfcc_scp = 'mfcc/raw_mfcc_train.1.scp'

utt_ids = []
vads = []
utt_id2vad = {}

with ReadHelper('scp:' + vad_scp) as vad_reader:
    for i, (utt_id, vad) in enumerate(vad_reader):
        if i < 10:
            utt_id2vad[utt_id] = vad
            print(utt_id, vad.shape)
        else:
            break
#print('vad reading completed')
#with ReadHelper('scp:' + mfcc_scp) as mfcc_reader:
#    for i, (utt_id, mfcc) in enumerate(mfcc_reader):
#        if utt_id in utt_id2vad.keys():
#            vad = utt_id2vad[utt_id]
#            assert(len(vad) == len(mfcc))
#            print(i, 'check',  vad.shape, mfcc.shape)
