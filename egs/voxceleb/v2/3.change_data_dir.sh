#!/usr/bin/env bash

nj=12
mfccdir=./mfcc

for name in train voxceleb1_test; do
    data=./data/$name
        # concatenate the .scp files together.
        for n in $(seq $nj); do
          cat $mfccdir/vad_$name.$n.scp || exit 1
        done > $data/vad.scp || exit 1
done

./3.1.make_train_aug_1m_vad_scp.py ./data/train_aug_1m/feats.scp  >./data/train_aug_1m/vad.scp
./3.2.write_utt2num_frames.py ./data/train_aug_1m/vad.scp ./data/train_aug_1m/utt2num_frames

for name in train train_aug_1m voxceleb1_test; do
    data=./data/$name
        # concatenate the .scp files together.
        for n in $(seq $nj); do
          cat $mfccdir/raw_mfcc_$name.$n.scp || exit 1
        done > $data/feats.scp || exit 1

    utils/split_data.sh $data $nj || exit 1;
    #chown ubuntu:ubuntu  ./data/$name/frame_shift
    echo 0.001 >./data/$name/frame_shift
done


