NIPA starthon: speech verification using Kaldi voxceleb recipe
==============================================================

egs/voxceleb에 위치하고 있는 voxceleb recipe를
기존의 FFT기반 스펙트로그램을 브레인소프트의 고해상도 스펙트로그램으로 대체하여 수행해본다.

egs/voxceleb에는 v1과 v2가 있는데, 
v1는 GMM-HMM(i-vector)기반이고 v2가 딥러닝(x-vector)를 이용한 것이다.
v2에 대한 참조논문은 다음과 같다. http://www.danielpovey.com/files/2018_icassp_xvectors.pdf 

v2의 default로 작동시키기
-------------------------
1. kaldi docker를 실행한다
2. ./cmd.sh를 열어서 export train_cmd="run.pl --mem 4G"로 고쳐준다. run.pl에 대한 사용법은 다음과 같다
./run.pl JOB=1:$nj 어쩌구log파일경로prefix.JOB.log 해당명령어
이렇게 하면 해당명령어를 $nj개의 cpu가 pararrel하게 실행하면서 log파일에 각각 쓴다
3. ./run.sh를 열어서 처음에 나오는 path들을 모두 각자의 상황에 맞게 고쳐준다. 
예를들어 나는 다음과 같이 고쳤다
dataset_root="/media/sangjik/hdd2"
voxceleb1_trials=data/voxceleb1_test/trials
voxceleb1_root=$dataset_root/dataset/speech/English/VoxCeleb1
voxceleb2_root=$dataset_root/dataset/speech/English/VoxCeleb2
nnet_dir="$dataset_root/speaker_verification/kaldi/xvector_nnet_1a.total_djt"
musan_root=$dataset_root/dataset/sound/musan
num_cpu=$(cat /proc/cpuinfo | awk '/^processor/{print $3}' | wc -l)
4.local/nnet3/xvector/run_xvector.sh파일에 거의 맨 밑에 train_raw_dnn.py의 옵션을 수정 --use-gpu=wait

run.sh의 구조
-------------
stage 0 ~ stage 12까지 나뉘어져 있다.
0. dataset을 준비하고 읽어서 indexing하는 부분이다 (./data/train과 ./data/voxceleb1_test이 생긴다)
1. MFCC를 추출하고 VAD를 생성한다
stage1을 끝내면 ./data디렉에는 conf  feats.scp  frame_shift  split8  utt2dur  utt2num_frames  vad.scp  이 새로생김
  steps/make_mfcc.sh 가 MFCC추출하는 모듈이고 (이 모듈을 후에 대체할 것이다)
  sid/compute_vad_decision.sh 이 VAD추출하는 모듈이다
2. stage2는 data augmentation을 하는 부분이다 (./data/train_aug 디렉토리가 생성된다)
3. stage2에서 augmentation을 통해서 생성된 음원의 MFCC를 추출하고 VAD를 추출한다.
default augmentation size는 1백만이다. 그래서 ./data/train_aug_1m을 생성한다 (1백만개만 임의로 선택)
그리고 ./data/train과 ./data/train_aug_1m을 합쳐서 ./data/train_combined를 생성한다.
4. stage4에서는 xvector훈련을 위한 example을 생성한다
더 자세히 이야기하면, CMVN(Cepstral Mean and Variance Normalization)을 적용하고, 위에서 생성한 VAD결과로 인하여 음성부분만 선택한다.
그래서 ./data/train_combined_no_sil을 생성한다.
5. stage5까지 하면 training example생성이 완료된다.
6. nnet examples 생성
7. neural net 아키텍쳐 설정
8. train 시작
9. x-vector 추출 
10. PLDA모델을 위한 전처리 및 PLDA모델 훈련
11. 임베딩벡터 plda scoring
12. 성과지표(Equal Error Rate:EER 및 min DCF1과 min DCF2) 계산
