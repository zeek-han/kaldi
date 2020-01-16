#!/usr/bin/env python3

import sys
from glob import glob


mfcc_scp = sys.argv[1]

utt_id2aug_id = {}
with open(mfcc_scp, 'r') as mfcc_scp_fd:
    for line in mfcc_scp_fd:
        utt_id_suffix, _ = line[:-1].split()
        utt_id2aug_id['-'.join(utt_id_suffix.split('-')[:-1])] = utt_id_suffix

for train_vad_scp in glob('./mfcc/vad_train.*.scp'):
    with open(train_vad_scp, 'r') as train_vad_fd:
        for line in train_vad_fd:
            utt_id, vad_content = line[:-1].split()
            if utt_id in utt_id2aug_id.keys():
                print(utt_id2aug_id[utt_id], vad_content)
                #print(utt_id, utt_id2aug_id[utt_id], vad_content)
