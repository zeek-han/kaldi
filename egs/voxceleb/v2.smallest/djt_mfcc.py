#!/usr/bin/env python3

import sys, os
from os.path import basename, dirname, join as p_join
from glob import glob
import tempfile
import subprocess
from functools import partial
from multiprocessing import cpu_count, Pool

import numpy as np
from kaldiio import WriteHelper
import scipy.io.wavfile as wav_file
from pydub import AudioSegment
from python_speech_features import mfcc as psf_mfcc
import librosa
import soundfile as sf

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def resample(wav, new_wav_dir='./'):
    target_sampling_rate = 384000
    data, sample_rate = sf.read(wav)
    if sample_rate != target_sampling_rate:
        super_samples = librosa.resample(data.T, sample_rate, target_sampling_rate)
        output_path = p_join(new_wav_dir, basename(wav))
        sf.write(output_path, super_samples.T, target_sampling_rate)
        print(output_path)

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

    ## get mfcc
    with tempfile.TemporaryDirectory() as new_wav_dir:
        this_resample = partial(resample, new_wav_dir=new_wav_dir)
        num_of_cpus = cpu_count()
        with Pool(num_of_cpus, maxtasksperchild=5) as p:
            p.map(this_resample, wavs, chunksize=5)
        #resample(wavs, new_wav_dir)
        for resampled_wav in glob(p_join(new_wav_dir, '*.wav')):
            os.system('../../../../deejaytransform-sound_length_no_limit/djt freq -t 4  \
                      -f {} -sd -spt 0 -gpu {}'.format(resampled_wav, 0))
        subprocess.call(['./djs2mfcc.py', new_wav_dir, input_wav_dir])
    with open(utt2dur_path, 'w') as fd1:
        print('\n'.join(utt_durs), file=fd1)
    with open(utt2num_frames_path, 'w') as fd2:
        print('\n'.join(utt_num_frames), file=fd2)
