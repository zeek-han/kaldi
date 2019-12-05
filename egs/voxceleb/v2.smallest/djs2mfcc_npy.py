#!/usr/bin/env python3

import struct
from glob import glob
from functools import partial
from multiprocessing import cpu_count, Pool
import sys
import os
from os.path import dirname, basename, join as p_join

import scipy
import numpy as np
import librosa

def getDtype(inputformat):
    if inputformat == 0:
        return np.int16
    else:
        return np.float32

def loadDJSpectrogram(filePath, inputformat):
    try:
        with open(filePath, "rb") as f:
            header = f.read(32)
            numOfChannels, lowestFreq, highestFreq, numOfSpectrums = \
                                                    struct.unpack('iiii', header[:16])
            data = np.fromfile(f, dtype=getDtype(inputformat))

    except IOError as e:
        print("Couldn't open file (%s.)" % e)
        return 0, 0, 0, 0, []

    spectrogram = data.reshape(highestFreq - lowestFreq + 1, numOfSpectrums)
    return numOfChannels, lowestFreq, highestFreq, numOfSpectrums, spectrogram

def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)

def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)


def get_filterbanks(nfilt=20,samplerate=16000,lowfreq=0,highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the sample rate of the signal we are working with, in Hz. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel,highmel,nfilt+2)
    bin = mel2hz(melpoints).astype(int) - lowfreq
    bin[0] = 0
    bin[-1] = int(highfreq - lowfreq)
    fbank = np.zeros([nfilt, (highfreq - lowfreq + 1)])
    for i in range(0, nfilt):
        for j in range(bin[i], bin[i + 1]):
            fbank[i, j] = (j - bin[i]) / (bin[i + 1] - bin[i])
        for j in range(bin[i + 1], bin[i + 2]):
            fbank[i, j] = 1 - (j - bin[i + 1])/ (bin[i + 2] - bin[i + 1])
    return fbank

def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes,ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L/2.)*np.sin(np.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra

def fbank(power_spectrogram, n_mels, sample_rate, fmin, fmax):
    filterbanks = get_filterbanks(nfilt=n_mels, samplerate=sample_rate, 
                                  lowfreq=fmin, highfreq=fmax)
    feat = np.dot(power_spectrogram, filterbanks.T) # compute the filterbank energies
    feat = np.where(feat == 0,np.finfo(float).eps,feat) # if feat is zero, we get problems with log
    energy = np.sum(power_spectrogram, 1)
    # if energy is zero, we get problems with log
    energy = np.where(energy == 0,np.finfo(float).eps,energy) 
    return feat, energy

def power_spectrogram(djs, lower_bound_freq, upper_bound_freq):
    num_of_channels, lowest_freq, highest_freq, num_of_spectrums,  \
                                                       spectrogram = loadDJSpectrogram(djs, 1)
    assert lower_bound_freq >= lowest_freq
    assert highest_freq >= upper_bound_freq 
    #Transpose to use python_speech_features library
    return np.square(spectrogram[(lowest_freq - lower_bound_freq): \
                                 (upper_bound_freq - highest_freq)]).T

def mfcc(power_spectrogram, n_mfcc=20, n_mels=128, dct_type=2, norm='ortho', 
         lowfreq=50, highfreq=8000, ceplifter=22, appendEnergy=True):
    feat, energy = fbank(power_spectrogram, n_mels, 16000, lowfreq, highfreq)
    feat = np.log(feat)
    feat = scipy.fftpack.dct(feat, type=dct_type, axis=1, norm=norm)[:,:n_mfcc]
    feat = lifter(feat,ceplifter)
    if appendEnergy: feat[:,0] = np.log(energy) # replace first cepstral coefficient with log of frame energy
    return feat

def save_mfcc(djs, output_dir='./'):
    #mfcc(djs)
    n_mfcc = 30
    n_mels = 30
    lowest_freq = 50
    highest_freq = 7600


    # mfcc from DJS
    djspec_square = power_spectrogram(djs, lowest_freq, highest_freq)
    mfcc_from_djs = mfcc(djspec_square, n_mfcc=n_mfcc, n_mels=n_mels, 
                         lowfreq=lowest_freq, highfreq=highest_freq)
    mfcc_djs_path = p_join(output_dir, basename(djs)[:-10] + '.npy')
    os.makedirs(dirname(mfcc_djs_path), exist_ok=True)
    np.save(mfcc_djs_path, mfcc_from_djs)
    print(mfcc_djs_path)


