#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc


# The trials file is downloaded by local/make_voxceleb1_v2.pl.
dataset_root="/media/sangjik/hdd2"
voxceleb1_trials=data/voxceleb1_test/trials
voxceleb1_root=$dataset_root/dataset/speech/English/VoxCeleb1
voxceleb2_root=$dataset_root/dataset/speech/English/VoxCeleb2
nnet_dir="/home/sangjik/speaker_verification/kaldi/xvector_nnet_1a.total_dataset_djt"
musan_root=$dataset_root/dataset/sound/musan
num_cpu=`cat /proc/cpuinfo | awk '/^processor/{print $3}' | wc -l`

stage=0
echo stage_3   `date`
if [ $stage -le 3 ]; then
  echo stage_3.1   `date`
  # Take a random subset of the augmentations
  utils/subset_data_dir.sh data/train_aug 50 data/train_aug_1m
  utils/fix_data_dir.sh data/train_aug_1m

  echo stage_3.2  `date`
  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc_zeek_for_substitution2.sh --mfcc-config conf/mfcc.conf --nj $num_cpu --cmd "$train_cmd" \
    data/train_aug_1m exp/make_mfcc $mfccdir

  echo stage_3.3  `date`
  # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/train_combined data/train_aug_1m data/train
fi


