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

def gpu_count():
    with tempfile.NamedTemporaryFile() as tmp_fd:
        os.system("nvidia-smi | grep +-------------------------------+----------------------+----------------------+ | wc -l >{}".format(tmp_fd.name))
        with open(tmp_fd.name, 'r') as fd:
            for line in fd:
                num_gpu = int(line[:-1])
    return num_gpu

def divide(seq, divisor):
    len_divided = int(len(seq) / divisor)
    for i in range(divisor - 1):    
        yield i, seq[(i * len_divided):((i + 1) * len_divided)]
    yield (i + 1), seq[((divisor - 1) * len_divided):]

def wav2djs(gpuid_resampledwavs):
    gpu_id, resampled_wavs = gpuid_resampledwavs
    for resampled_wav in resampled_wavs:
        if len(AudioSegment.from_wav(resampled_wav)) < 150000:
            #os.system('../../../../deejaytransform-sound_length_no_limit/djt freq \
            #           -t 4 -f {} -sd -spt 0 -gpu {}'.format(resampled_wav, gpu_id))
            os.system('./for_loop')
        else:
            print("Too long wav:", resampled_wav)
            raise ValueError

if __name__ == '__main__':
    input_prefix = sys.argv[1]
    input_wav_dir = sys.argv[2]
   
    num = input_wav_dir.split('.')[-1] 
    utt2dur_path = p_join(dirname(input_prefix), 'utt2dur.' + num)
    utt2num_frames_path = p_join(dirname(input_prefix), 'utt2num_frames.' + num)
   
    utt_durs = [] 
    utt_num_frames = []
    utt2dur = {}
    utt2num_frames = {}
    wavs = glob(p_join(input_wav_dir, '*.wav'))
    for wav in wavs:
        utt_id = basename(wav[:-4])
        wav_len = len(AudioSegment.from_wav(wav))
        num_frames = int(wav_len / 10)
        dur = wav_len / 1000
        utt_durs.append('{} {}'.format(utt_id, dur))
        utt2dur[utt_id] = dur
        utt_num_frames.append('{} {}'.format(utt_id, num_frames))
        utt2num_frames[utt_id] = num_frames

    ## get mfcc
    with tempfile.TemporaryDirectory() as new_wav_dir:
        this_resample = partial(resample, new_wav_dir=new_wav_dir)
        num_of_cpus = cpu_count()
        with Pool(num_of_cpus, maxtasksperchild=20) as p:
            p.map(this_resample, wavs, chunksize=5)
        num_of_gpus = gpu_count()
        new_wavs = glob(p_join(new_wav_dir, '*.wav'))
        divided_new_wavs = list(divide(new_wavs, num_of_gpus))
##start
        tmp_written_dir =  './tmp_written/' + input_wav_dir
        os.makedirs(tmp_written_dir, exist_ok=True) 
        for divided_new_wav in divided_new_wavs:
            gpu_id, wavs = divided_new_wav
            with open(p_join(tmp_written_dir, str(gpu_id)), 'w') as ffd:
                print(len(wavs), wavs, file=ffd)
##end
        #with Pool(num_of_gpus, maxtasksperchild=num_of_gpus) as p:
        with Pool(5, maxtasksperchild=num_of_gpus) as p:
            p.map(wav2djs, divided_new_wavs, chunksize=num_of_gpus)

        subprocess.call(['./djs2mfcc.py', new_wav_dir, input_wav_kir])
    with open(utt2dur_path, 'w') as fd1:
        #print('\n'.join(utt_durs), file=fd1)
        print('\n'.join(['{} {}'.format(k, v) for (k, v) in utt2dur.items()]), file=fd1)
    with open(utt2num_frames_path, 'w') as fd2:
        #print('\n'.join(utt_num_frames), file=fd2)
        print('\n'.join(['{} {}'.format(k, v) for (k, v) in utt2num_frames.items()]), file=fd2)
