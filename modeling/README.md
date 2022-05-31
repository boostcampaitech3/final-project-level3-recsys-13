# final_project_modeling_base
![](https://velog.velcdn.com/images/whattsup_kim/post/d5bee838-179a-48e8-bec2-04829cf2aab1/image.png)

## How To Use
### setting
- (OpenBLAS Warning이 나오는 경우)
    - 쉘에 `export OPENBLAS_NUM_THREADS=1`
- CUDA 설치
    - `conda config --append channels conda-forge`
    - `conda install cudatoolkit=11.2 -c conda-forge`
    - `conda install cudatoolkit-dev=11.2 -c conda-forge`
- 'modeling/core/에 db관련 키 파일 넣기
    - .env : db연결을 위한 key
    - storage.json : storage 연결을 위한 key
- 'modeling/core/에 storage 관련 json파일 넣기
    - storage.json
- shell에 `wandb init`
    - API key 입력
    - 원하는 디렉토리 선택 :'foodcom'

### Train  
- 모델 학습 커맨드
    - `python train.py`
- 학습 가능한 모델
    - als: `--model als`
    - bert2vec: `--model bert2vec`
    - ...
- argparse 사용법(subparse)
    - argparse는 모든 모델에서 사용하는 공동 argparse와 하위 각 모델에서 사용할 subparse로 구성됨
    - 예시1) als를 사용하고 싶을 때
        - `python train.py als`
    - 예시2) bert2vec에서 공동 argparse인 'seed'와 bert2vec subparse인 'top_n'를 사용하고 싶을 때
        - `python train.py --seed 777 bert2vec --top_n 30`
- train에서는 모델 학습 및 wandb 로그 저장만 진행되며, inference는 따로 구현되어 있음.

### Inference  
- inference 커맨드 : train과 같음
- infernce.py는 모델을 모든 데이터로 학습시키고, 결과를 google storage로 전송시킵니다.

## TODO
- torch 모델 추가
- airflow: 배치 단위 학습  
: 데이터 업데이트 시 쉘 커맨드 등을 활용하여 모든 모델 train(및 기록) 맟 inference.