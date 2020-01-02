#!/usr/bin/env python3

import sys
from glob import iglob, glob
from os.path import basename, dirname, join as p_join
import json

import numpy as np
from kaldiio import ReadHelper, WriteHelper


if __name__ == '__main__':

    utt_id2num_frames = {}
    for name in ['train', 'voxceleb1_test']:
        for i, vad_scp in enumerate(glob('./mfcc/vad_{}.*.scp'.format(name))):
            vad_ark = vad_scp[:-4] + '.ark'
            with ReadHelper('scp:' + vad_scp) as reader:
                for utt_id, vad in reader:
                    utt_id2num_frames[utt_id] = len(vad)


        with open('./data/{}/utt2num_frames'.format(name), 'w') as fd:
            for k, v in sorted(utt_id2num_frames.items()):
                print(k, v, file=fd)
