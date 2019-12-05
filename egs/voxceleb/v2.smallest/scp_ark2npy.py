#!/usr/bin/env python

import numpy as np
import kaldiio

id2mfcc = kaldiio.load_scp('/home/sangjik/kaldi/egs/voxceleb/v2.smallest/mfcc/raw_mfcc_train.10.scp')
for utt_id, mfcc in id2mfcc.items():
    #print(utt_id, mfcc.shape)
    np.save('./tmp_mfcc/{}.npy'.format(utt_id), mfcc)
