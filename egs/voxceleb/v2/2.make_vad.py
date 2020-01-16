#!/usr/bin/env python3

import os
import sys
from glob import iglob, glob
from os.path import basename, dirname, join as p_join
import json

import numpy as np
from kaldiio import ReadHelper, WriteHelper

def make_vad_scp_in_mfcc_scp(input_dir):
    pass

if __name__ == '__main__':
    input_mfcc_dir = './mfcc.bk'
    mfcc_scp_dir = './mfcc_scp'
#    for scp in sorted(glob(p_join(input_mfcc_dir, 'vad_*.scp'))):
#        new_scp = p_join(mfcc_scp_dir, basename(scp))
#        print(scp, new_scp)
#        with open(new_scp, 'w') as new_fd:
#            with open(scp, 'r') as fd:
#                for line in fd:
#                    print(line[:-1].replace('/mfcc/', '/{}/'.format(input_mfcc_dir)),
#                           file=new_fd)

    utt_id2vad = {}
    for scp in sorted(glob(p_join(input_mfcc_dir, 'vad_*.scp'))):
        with ReadHelper('scp:' + scp) as reader:
            for utt_id, arr in reader:
                utt_id2vad[utt_id] = arr
                #print(utt_id, utt_id2high_resol_len[utt_id], len(arr))
    #for k, v in utt_id2vad.items():
    #    print(k, v.shape)
    ##print(len(utt_id2vad.keys()))

#################3 이게 mfcc가 나와야 vad를 만들 수 있다. shape을 그것에 맞추어야 하니까
#    os.system('rm {}/raw_mfcc*.scp'.format(mfcc_scp_dir))
#    os.system('cp ./mfcc/raw_mfcc* {}'.format(mfcc_scp_dir))
################# mfcc_scp안에 mfcc안에 있는 scp를 죄다 넣어야 한다
    for name in ['train', 'voxceleb1_test']:
        for i, mfcc in enumerate(glob(p_join(os.getcwd(), 'mfcc', 'raw_mfcc_{}.*.scp'.format(name)))):
            #with open(mfcc_scp, 'r') as fd:
            #    for line in fd:
            #        print(line[:-1])
            vad_scp = mfcc.replace('/raw_mfcc_', '/vad_')
            vad_ark = vad_scp[:-4] + '.ark'
            with ReadHelper('scp:' + mfcc) as reader:
                with WriteHelper('ark,scp:{},{}'.format(vad_ark, vad_scp)) as writer:
                    for utt_id, high_resolution_mfcc in reader:
                        #print(utt_id, high_resolution_mfcc.shape)
                        try:
                            #low_resolution_vad = np.load(utt_id2vad[utt_id])
                            low_resolution_vad = utt_id2vad[utt_id]
                            assert isinstance(low_resolution_vad, np.ndarray)
                            high_resoultion_vad = np.zeros(len(high_resolution_mfcc),
                                                           dtype=low_resolution_vad.dtype)
                            for i in range(len(low_resolution_vad)):
                                from_idx = 10 * i
                                to_idx = min(from_idx + 10, len(high_resoultion_vad))
                                high_resoultion_vad[from_idx:to_idx] = low_resolution_vad[i]
                            writer[utt_id] = high_resoultion_vad
                        except KeyError:
                            print("KeyError:", utt_id, utt_id2vad[utt_id].shape)
                        except AssertionError:
                            print("AssertionError:", utt_id, type(low_resolution_vad))


