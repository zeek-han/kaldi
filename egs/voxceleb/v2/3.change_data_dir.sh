#!/usr/bin/env bash

nj=12
mfccdir=./mfcc

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

for name in train voxceleb1_test; do
    data=./data/$name
        # concatenate the .scp files together.
        for n in $(seq $nj); do
          cat $mfccdir/vad_$name.$n.scp || exit 1
        done > $data/vad.scp || exit 1
done

