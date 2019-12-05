#!/usr/bin/env python3

import sys, os
from os.path import basename, dirname, join as p_join
from glob import glob

import numpy as np
from kaldiio import WriteHelper
import scipy.io.wavfile as wav_file
from pydub import AudioSegment
from python_speech_features import mfcc as psf_mfcc

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

if __name__ == '__main__':
    input_prefix = sys.argv[1]
    input_wav_dir = sys.argv[2]
   
    num = input_wav_dir.split('.')[-1] 
    utt2dur_path = p_join(dirname(input_prefix), 'utt2dur.' + num)
    utt2num_frames_path = p_join(dirname(input_prefix), 'utt2num_frames.' + num)
   
    utt_durs = [] 
    utt_num_frames = []
    wavs = glob(p_join(input_wav_dir, '*.wav'))
    for wav in wavs:
        utt_id = basename(wav[:-4])
        wav_len = len(AudioSegment.from_wav(wav))
        num_frames = int(wav_len / 10)
        dur = wav_len / 1000
        utt_durs.append('{} {}'.format(utt_id, dur))
        utt_num_frames.append('{} {}'.format(utt_id, num_frames))

        # get mfcc
        sr, samples = wav_file.read(wav)
        psf_mfcc_feat = psf_mfcc(samples, numcep=30, nfilt=30, nfft=640, lowfreq=20, highfreq=8000)
        np.save(wav[:-4] + '.npy', psf_mfcc_feat)
        try:
            os.remove(wav)
        except OSError:
            pass
    print(utt2dur_path)
    try:
        os.remove(utt2dur_path)
        os.remove(utt2num_frames_path)
    except OSError:
        pass
    with open(utt2dur_path, 'w') as fd1:
        print('\n'.join(utt_durs), file=fd1)
    print('\n'.join(utt_durs))
    #print('file_len(utt2dur)=', file_len(utt2dur_path), len(utt_durs), utt_durs[0], utt_durs[-1])
    with open(utt2num_frames_path, 'w') as fd2:
        print('\n'.join(utt_num_frames), file=fd2)
