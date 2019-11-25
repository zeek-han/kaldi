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
voxceleb1_trials=data/voxceleb1_test/trials
voxceleb1_root=/media/sangjik/hdd2/dataset/speech/English/VoxCeleb1
voxceleb2_root=/media/sangjik/hdd2/dataset/speech/English/VoxCeleb2
nnet_dir="/media/sangjik/hdd2/speaker_verification/kaldi/xvector_nnet_1a"
musan_root=/media/sangjik/hdd2/dataset/sound/musan

stage=0

# 1~2분이면 끝나더라
if [ $stage -le 0 ]; then
  #local/make_voxceleb2.pl $voxceleb2_root dev data/voxceleb2_train
  #local/make_voxceleb2.pl $voxceleb2_root test data/voxceleb2_test
  ## This script creates data/voxceleb1_test and data/voxceleb1_train for latest version of VoxCeleb1.
  ## Our evaluation set is the test portion of VoxCeleb1.
  #local/make_voxceleb1_v2.pl $voxceleb1_root dev data/voxceleb1_train
  #local/make_voxceleb1_v2.pl $voxceleb1_root test data/voxceleb1_test
  ## if you downloaded the dataset soon after it was released, you will want to use the make_voxceleb1.pl script instead.
  ## local/make_voxceleb1.pl $voxceleb1_root data
  ## We'll train on all of VoxCeleb2, plus the training portion of VoxCeleb1.
  ## This should give 7,323 speakers and 1,276,888 utterances.
#여기까지 하면 ./data디렉이 생기면서  data/train data/voxceleb2_train data/voxceleb2_test data/voxceleb1_train 이 만들어지고

  #utils/combine_data.sh data/train data/voxceleb2_train data/voxceleb2_test data/voxceleb1_train
#이걸하면 ./data/train 이 생긴다, 여기에는 wav.scp, spk2utt, utt2spk
#wav.scp에는 voxceleb2의 train과 test, 그리고  voxceleb1의 train까지만 들어있다
# 1,092,009(2_dev) + 36,237(2_test) + 148,462(1_dev) = 1,276,888 (wav.scp의 행 개수)
# utt2spk도 wc하면 1,276,888 (utterance path 와 id매칭시켜주는 파일)
# spk2utt는 7,323( 2_dev의 화자수 5994 +  1_dev의 1211 + 2_test의 118) 
# 근데 combine_data를 하고 나면, 워닝이 생기더라

# wav.scp와 utt2spk는 utils/combine_data.sh의 111~134번째 줄에서 직접 combine한 결과다
# spk2utt는 136번째 줄에서 spk2utt를 생성한다
############
# utils/combine_data.sh data/train data/voxceleb2_train data/voxceleb2_test data/voxceleb1_train
# utils/combine_data.sh [info]: not combining utt2uniq as it does not exist   	->이건 없어도 된단다, 다만 하나라도 있으면 모두에게 있어야한단다, 생성해야한단다
# utils/combine_data.sh [info]: not combining segments as it does not exist	->얘두 utt2uniq과 똑같이 없어도 되지만 하나라도 있다면 모두에게 있게 생성해야 한단다
# utils/combine_data.sh: combined utt2spk
# utils/combine_data.sh [info]: not combining utt2lang as it does not exist
# utils/combine_data.sh [info]: not combining utt2dur as it does not exist
# utils/combine_data.sh [info]: not combining utt2num_frames as it does not exist
# utils/combine_data.sh [info]: not combining reco2dur as it does not exist
# utils/combine_data.sh [info]: not combining feats.scp as it does not exist
# utils/combine_data.sh [info]: not combining text as it does not exist
# utils/combine_data.sh [info]: not combining cmvn.scp as it does not exist
# utils/combine_data.sh [info]: not combining vad.scp as it does not exist
# utils/combine_data.sh [info]: not combining reco2file_and_channel as it does not exist
# utils/combine_data.sh: combined wav.scp
# utils/combine_data.sh [info]: not combining spk2gender as it does not exist
# fix_data_dir.sh: kept all 1276888 utterances.
# fix_data_dir.sh: old files are kept in data/train/.backup
############
# fix_data_dir.sh에서 45~62줄에서 /.backup에 있는 파일들은 sort하기 전에 원래 있던 파일들을 백업하고, 그리고나서 sort해버린다
# fix_data_dir.sh에서 ...뭐 기본적으로 sort하고 filter하고 그런건데 sort는 알겠는데 
# fix_data_dir.sh에서 filter는 utils/filter_scp.pl이 기본인것 같은데, 얘는 그냥 speaker-id리스트와 utt-id리스트를 받아서 utt중에 speaker-id에 있는애들만 뽑아내는것 같다
fi

#if [ $stage -le 1 ]; then
#  # Make MFCCs and compute the energy-based VAD for each dataset
#  for name in train voxceleb1_test; do
#    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
#      data/${name} exp/make_mfcc $mfccdir
#########steps/make_mfcc.sh를 하면 ./mfcc에 raw_mfcc_${name}.scp와 .ark가 생긴다.  이게 바로 mfcc
#    utils/fix_data_dir.sh data/${name}
#    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
#      data/${name} exp/make_vad $vaddir
#    utils/fix_data_dir.sh data/${name}
#  done
#fi

## In this section, we augment the VoxCeleb2 data with reverberation,
## noise, music, and babble, and combine it with the clean data.
#if [ $stage -le 2 ]; then
#  frame_shift=0.01
#  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train/utt2num_frames > data/train/reco2dur
#
#  if [ ! -d "RIRS_NOISES" ]; then
#    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
#    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
#    unzip rirs_noises.zip
#  fi
#
#  # Make a version with reverberated speech
#  rvb_opts=()
#  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
#  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
#
#  # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
#  # additive noise here.
#  steps/data/reverberate_data_dir.py \
#    "${rvb_opts[@]}" \
#    --speech-rvb-probability 1 \
#    --pointsource-noise-addition-probability 0 \
#    --isotropic-noise-addition-probability 0 \
#    --num-replications 1 \
#    --source-sampling-rate 16000 \
#    data/train data/train_reverb
#  cp -f data/train/vad.scp data/train_reverb/
#  utils/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
#  rm -rf data/train_reverb
#  mv data/train_reverb.new data/train_reverb
#
#  # Prepare the MUSAN corpus, which consists of music, speech, and noise
#  # suitable for augmentation.
#  steps/data/make_musan.sh --sampling-rate 16000 $musan_root data
#
#  # Get the duration of the MUSAN recordings.  This will be used by the
#  # script augment_data_dir.py.
#  for name in speech noise music; do
#    utils/data/get_utt2dur.sh data/musan_${name}
#    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
#  done
#
#  # Augment with musan_noise
#  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
#  # Augment with musan_music
#  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
#  # Augment with musan_speech
#  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train data/train_babble
#
#  # Combine reverb, noise, music, and babble into one directory.
#  utils/combine_data.sh data/train_aug data/train_reverb data/train_noise data/train_music data/train_babble
#fi
#
#if [ $stage -le 3 ]; then
#  # Take a random subset of the augmentations
#  utils/subset_data_dir.sh data/train_aug 1000000 data/train_aug_1m
#  utils/fix_data_dir.sh data/train_aug_1m
#
#  # Make MFCCs for the augmented data.  Note that we do not compute a new
#  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
#  # the list.
#  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
#    data/train_aug_1m exp/make_mfcc $mfccdir
#
#  # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
#  # double the size of the original clean list.
#  utils/combine_data.sh data/train_combined data/train_aug_1m data/train
###여기까지 하면 data/train_combined가 생성됨
#fi
#
## Now we prepare the features to generate examples for xvector training.
#if [ $stage -le 4 ]; then
#  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
#  # wasteful, as it roughly doubles the amount of training data on disk.  After
#  # creating training examples, this can be removed.
#  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
#    data/train_combined data/train_combined_no_sil exp/train_combined_no_sil
###위에 prepare_feats_for_egs.sh는 data/train_combined를 input으로 하고 data/train_combined_no_sil과 exp/블라블라를 생성함
#  utils/fix_data_dir.sh data/train_combined_no_sil
#fi
#
#if [ $stage -le 5 ]; then
#  # Now, we need to remove features that are too short after removing silence
#  # frames.  We want atleast 5s (500 frames) per utterance.
#  min_len=400
#  mv data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2num_frames.bak
#  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train_combined_no_sil/utt2num_frames.bak > data/train_combined_no_sil/utt2num_frames
#  utils/filter_scp.pl data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2spk > data/train_combined_no_sil/utt2spk.new
#  mv data/train_combined_no_sil/utt2spk.new data/train_combined_no_sil/utt2spk
#  utils/fix_data_dir.sh data/train_combined_no_sil
#
#  # We also want several utterances per speaker. Now we'll throw out speakers
#  # with fewer than 8 utterances.
#  min_num_utts=8
#  awk '{print $1, NF-1}' data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/spk2num
#  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/train_combined_no_sil/spk2num | utils/filter_scp.pl - data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/spk2utt.new
#  mv data/train_combined_no_sil/spk2utt.new data/train_combined_no_sil/spk2utt
#  utils/spk2utt_to_utt2spk.pl data/train_combined_no_sil/spk2utt > data/train_combined_no_sil/utt2spk
#
#  utils/filter_scp.pl data/train_combined_no_sil/utt2spk data/train_combined_no_sil/utt2num_frames > data/train_combined_no_sil/utt2num_frames.new
#  mv data/train_combined_no_sil/utt2num_frames.new data/train_combined_no_sil/utt2num_frames
#
#  # Now we're ready to create training examples.
#  utils/fix_data_dir.sh data/train_combined_no_sil
#fi
#
## Stages 6 through 8 are handled in run_xvector.sh
#local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage -1 \
#  --data data/train_combined_no_sil --nnet-dir $nnet_dir \
#  --egs-dir $nnet_dir/egs
#
#if [ $stage -le 9 ]; then
#  # Extract x-vectors for centering, LDA, and PLDA training.
#  echo stage9.1
#  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 20 \
#    $nnet_dir data/train \
#    $nnet_dir/xvectors_train
#
#  # Extract x-vectors used in the evaluation.
#  echo stage9.2
#  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 10 \
#    $nnet_dir data/voxceleb1_test \
#    $nnet_dir/xvectors_voxceleb1_test
#fi
#
#if [ $stage -le 10 ]; then
#  # Compute the mean vector for centering the evaluation xvectors.
#  $train_cmd $nnet_dir/xvectors_train/log/compute_mean.log \
#    ivector-mean scp:$nnet_dir/xvectors_train/xvector.scp \
#    $nnet_dir/xvectors_train/mean.vec || exit 1;
#
#  # This script uses LDA to decrease the dimensionality prior to PLDA.
#  lda_dim=200
#  $train_cmd $nnet_dir/xvectors_train/log/lda.log \
#    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
#    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train/xvector.scp ark:- |" \
#    ark:data/train/utt2spk $nnet_dir/xvectors_train/transform.mat || exit 1;
#
#  # Train the PLDA model.
#  $train_cmd $nnet_dir/xvectors_train/log/plda.log \
#    ivector-compute-plda ark:data/train/spk2utt \
#    "ark:ivector-subtract-global-mean scp:$nnet_dir/xvectors_train/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
#    $nnet_dir/xvectors_train/plda || exit 1;
#fi
#
#if [ $stage -le 11 ]; then
#  $train_cmd exp/scores/log/voxceleb1_test_scoring.log \
#    ivector-plda-scoring --normalize-length=true \
#    "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train/plda - |" \
#    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$nnet_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$nnet_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
#    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" exp/scores_voxceleb1_test || exit 1;
#fi
#
#if [ $stage -le 12 ]; then
#  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials exp/scores_voxceleb1_test) 2> /dev/null`
#  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
#  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
#  echo "EER: $eer%"
#  echo "minDCF(p-target=0.01): $mindcf1"
#  echo "minDCF(p-target=0.001): $mindcf2"
#  # EER: 3.128%
#  # minDCF(p-target=0.01): 0.3258
#  # minDCF(p-target=0.001): 0.5003
#  #
#  # For reference, here's the ivector system from ../v1:
#  # EER: 5.329%
#  # minDCF(p-target=0.01): 0.4933
#  # minDCF(p-target=0.001): 0.6168
#fi
