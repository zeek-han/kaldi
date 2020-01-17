NIPA starthon: speech verification using Kaldi voxceleb recipe
==============================================================

egs/voxceleb에 위치하고 있는 voxceleb recipe를
기존의 FFT기반 스펙트로그램을 브레인소프트의 고해상도 스펙트로그램으로 대체하여 수행해본다.

egs/voxceleb에는 v1과 v2가 있는데, 
v1는 GMM-HMM기반이고 v2가 딥러닝(x-vector)를 이용한 것이다.
v2에 대한 참조논문은 다음과 같다. http://www.danielpovey.com/files/2018_icassp_xvectors.pdf 

v2의 작동법은 다음과 같다
1. kaldi docker를 실행한다
2. ./cmd.sh를 열어서 export train_cmd="run.pl --mem 4G"로 고쳐준다
3.

