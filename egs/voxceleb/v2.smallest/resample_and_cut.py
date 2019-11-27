#!/usr/bin/env python3

import sys, os
from os.path import basename, dirname, join as p_join
from glob import glob
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count, Pool
import tempfile
from shutil import copyfile

from scipy.io import wavfile
from pydub import AudioSegment
import librosa
import soundfile as sf

def resample(wav, new_wav_dir='./'):
    target_sampling_rate = 384000
    data, sample_rate = sf.read(wav)
    output_path = p_join(new_wav_dir, basename(wav))
    if sample_rate != target_sampling_rate:
        super_samples = librosa.resample(data.T, sample_rate, target_sampling_rate)
        sf.write(output_path, super_samples.T, target_sampling_rate)
    else:
        copyfile(wav, output_path)
    print(output_path)

def gpu_count():
    with tempfile.NamedTemporaryFile() as tmp_fd:
        os.system("nvidia-smi | grep +-------------------------------+----------------------+----------------------+ | wc -l >{}".format(tmp_fd.name))
        with open(tmp_fd.name, 'r') as fd:
            for line in fd:
                num_gpu = int(line[:-1])
    return num_gpu
 
def cut_wav(long_wavs, output_dir, window_size, stride):
    os.makedirs(output_dir, exist_ok=True)
    for file_path in long_wavs:
        sr, samples = wavfile.read(file_path)
        for i in range(0, len(samples), stride * sr):
            sliced_samples = samples[i:min(i + window_size * sr, len(samples))]
            from_t = int(i / sr)
            to_t =  int(min(i + window_size * sr, len(samples)) / sr)
            sliced_wav_path = p_join(output_dir, '{}.{:03d}_{:03d}.wav'.format(basename(file_path)[:-4], from_t, to_t))
            wavfile.write(sliced_wav_path, sr, sliced_samples)
            # MUST PRINT THIS to resample this wav
            print(sliced_wav_path)
            if i + window_size * sr >= len(samples):
                break


if __name__ == '__main__':
    input_prefix = sys.argv[1]
    input_wav_dir = sys.argv[2]
    djs_dir = sys.argv[3]
    num_of_wav_chunk = int(sys.argv[4])
   
    num = input_wav_dir.split('.')[-1] 
    utt2dur_path = p_join(dirname(input_prefix), 'utt2dur.' + num)
    utt2num_frames_path = p_join(dirname(input_prefix), 'utt2num_frames.' + num)
   
    #utt_durs = [] 
    #utt_num_frames = []
    utt2dur = {}
    utt2num_frames = {}
    wavs = sorted(glob(p_join(input_wav_dir, '*.wav')))
    long_wavs = []
    #gpu_id = 0
    num_wav_per_chunk = max(int(len(wavs) / num_of_wav_chunk), 1)
    num_wav_per_gpu = max(int(num_wav_per_chunk / gpu_count()), 1)
    max_chunk_id = num_of_wav_chunk - 1
    max_gpu_id = gpu_count() - 1
    gpu_id2wavs = defaultdict(list)
    i = 0
    pre_chunk_id = -1
    num_wav_in_this_chunk = 0
    MAX_DURATION = 65
    for wav in wavs:
        utt_id = basename(wav[:-4])
        wav_len = len(AudioSegment.from_wav(wav))
        num_frames = int(wav_len / 10)
        dur = wav_len / 1000
        #utt_durs.append('{} {}'.format(utt_id, dur))
        utt2dur[utt_id] = dur
        #utt_num_frames.append('{} {}'.format(utt_id, num_frames))
        utt2num_frames[utt_id] = num_frames
        if dur < MAX_DURATION:
            i += 1
            wav_chunk_id = min(int(i / num_wav_per_chunk), max_chunk_id)
            if pre_chunk_id != wav_chunk_id:
                num_wav_in_this_chunk = 0
                pre_chunk_id = wav_chunk_id
            num_wav_in_this_chunk += 1
            gpu_id = min(int(num_wav_in_this_chunk / num_wav_per_gpu), max_gpu_id)
            gpu_id2wavs[(wav_chunk_id, gpu_id)].append(wav)
        else:
            long_wavs.append(wav)

    with open(utt2dur_path, 'w') as fd1:
        print('\n'.join(['{} {}'.format(k, v) for (k, v) in utt2dur.items()]), file=fd1)
    with open(utt2num_frames_path, 'w') as fd2:
        print('\n'.join(['{} {}'.format(k, v) for (k, v) in utt2num_frames.items()]), file=fd2)

    for (wav_chunk_id, gpu_id), short_wavs in gpu_id2wavs.items():
        print((wav_chunk_id, gpu_id))
    for (wav_chunk_id, gpu_id), short_wavs in gpu_id2wavs.items():
        resampled_dir = p_join(djs_dir, 'chunk_{}'.format(wav_chunk_id), 'gpu_{}'.format(gpu_id))
        os.makedirs(resampled_dir, exist_ok=True)
        this_resample = partial(resample, new_wav_dir=resampled_dir)
        num_of_cpus = cpu_count()
        with Pool(num_of_cpus, maxtasksperchild=20) as p:
            p.map(this_resample, short_wavs, chunksize=5) 
    # long_wavs는 잘라서 넣어준다
    cutted_dir = p_join(djs_dir, 'long_wav.before_resampled')
    cut_wav(long_wavs, cutted_dir, MAX_DURATION, MAX_DURATION - 1)
    cutted_resampled_dir = p_join(djs_dir, 'long_wav')
    os.makedirs(cutted_resampled_dir, exist_ok=True)
    for cutted_wav in glob(p_join(cutted_dir, '*.wav')):
        resample(cutted_wav, cutted_resampled_dir)
