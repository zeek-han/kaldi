#!/usr/bin/env python3

import sys
from glob import iglob, glob
from os.path import basename, dirname, join as p_join
import json

import numpy as np
from kaldiio import ReadHelper, WriteHelper


def mfcc_path2utt_id(mfcc_path):
    return basename(mfcc_path)[:-4]

if __name__ == '__main__':

    #npy_high_resolution_root = '/home/tmp_sangjik/vad_npy/only_Voxceleb1'
    npy_high_resolution_root  = '/media/sangjik/hdd2/dataset/speech/English/VoxCeleb1/vad_npy'
    #for name in ['train', 'voxceleb1_test']:
    #    utt_id2num_frames = {}
    #    for vad_scp in sorted(glob('/home/ubuntu/mfcc.output/vad_{}.*.scp'.format(name))):
    #        #print(vad_scp)
    #        with ReadHelper('scp:' + vad_scp) as reader:
    #            for utt_id, low_resolution_vad in reader:
    #                vad_path = p_join(npy_high_resolution_root, utt_id + '.npy')
    #                np.save(vad_path, low_resolution_vad)
               

        #for mfcc_path in iglob(p_join(npy_high_resolution_root, name + "*", "*.npy"), 
        #                       recursive=True):
        #    utt_id = mfcc_path2utt_id(mfcc_path)
        #    utt_id2num_frames = len(np.load(mfcc_path))
        ##print(len(utt_id2num_frames), utt_id2num_frames[0])

        #with open(p_join('./data', name, 'utt2num_frames'), 'w') as utt2num_frames:
        #    print('\n'.join(["{} {}".format(k, v) for k, v in utt_id2num_frames.items()]), 
        #          file=utt2num_frames)
        #
        #with open(p_join('./data', name, 'utt2num_frames.json'), 'w') as fp:
        #    json.dump(utt_id2num_frames, fp)

        

    




    #with open('/home/sangjik/utt_id2vad.json', 'r') as fd:
    #    utt_id2vad = json.load(fd)

    for name in ['train', 'voxceleb1_test']:
        for i, mfcc_scp in enumerate(glob('./mfcc/raw_mfcc_{}.*.scp'.format(name))):
            vad_scp = mfcc_scp.replace('/raw_mfcc_', '/vad_')
            vad_ark = vad_scp[:-4] + '.ark'
            with ReadHelper('scp:' + mfcc_scp) as reader:
                with WriteHelper('ark,scp:{},{}'.format(vad_ark, vad_scp)) as writer:
                    for utt_id, high_resolution_mfcc in reader:
                        low_resolution_vad = np.load(p_join(npy_high_resolution_root, utt_id + '.npy'))
                        high_resoultion_vad = np.zeros(len(high_resolution_mfcc), 
                                                       dtype=low_resolution_vad.dtype)
                        for i in range(len(low_resolution_vad)):
                            from_idx = 10 * i
                            to_idx = min(from_idx + 10, len(high_resoultion_vad))
                            high_resoultion_vad[from_idx:to_idx] = low_resolution_vad[i]
                        writer[utt_id] = high_resoultion_vad

