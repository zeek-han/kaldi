#!/usr/bin/env python3

import os
import sys
from glob import iglob, glob
from os.path import basename, dirname, join as p_join
import json

import numpy as np
from kaldiio import ReadHelper, WriteHelper


vad_scp = sys.argv[1]
utt2num_frames = sys.argv[2]

utt_ids = []
vads = []
utt_id2vad = {}

with ReadHelper('scp:' + vad_scp) as vad_reader:
    #for i, (utt_id, vad) in enumerate(vad_reader):
    #    utt_id2vad[utt_id] = vad
    #    print(utt_id, len(vad))
    with open(utt2num_frames, 'w') as writer:
        for i, (utt_id, vad) in enumerate(vad_reader):
            utt_id2vad[utt_id] = vad
            print(utt_id, len(vad), file=writer)

#print('vad reading completed')
#with ReadHelper('scp:' + mfcc_scp) as mfcc_reader:
#    for i, (utt_id, mfcc) in enumerate(mfcc_reader):
#        if utt_id in utt_id2vad.keys():
#            vad = utt_id2vad[utt_id]
#            assert(len(vad) == len(mfcc))
#            print(i, 'check',  vad.shape, mfcc.shape)
