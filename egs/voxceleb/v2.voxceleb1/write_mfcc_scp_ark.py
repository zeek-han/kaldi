#!/usr/bin/env python3

import sys, os
from os.path import basename, dirname, join as p_join
from glob import glob

import numpy as np
from kaldiio import WriteHelper
import scipy.io.wavfile as wav_file
from python_speech_features import mfcc as psf_mfcc

if __name__ == '__main__':
    input_mfcc_dir= sys.argv[1]
    output_prefix = sys.argv[2]

    mfccs = glob(p_join(input_mfcc_dir, '*.npy'))
    os.makedirs(dirname(output_prefix), exist_ok=True)
    new_scp = output_prefix + '.scp'
    new_ark = output_prefix + '.ark'
    with WriteHelper('ark,scp:{},{}'.format(new_ark, new_scp)) as writer:
        for mfcc in mfccs:
            utt_id = basename(mfcc)[:-4]
            writer[utt_id] = np.load(mfcc)
