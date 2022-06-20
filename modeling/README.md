
## How To Use
### setting
- 쉘에 다음 커맨드를 입력하여 make를 설치해주세요
    - `apt-get install gcc make`
- 새로운 가상환경을 만들고, 진입해주세요.
    - `conda create -n {가상환경이름} python=3.8`
    - `conda activate {가상환경이름}`  
    (or `source activate {가상환경이름}`)
- final-project-level3-recsys-13/modeling 폴더로 진입하여 다음 쉘 명령어를 입력해주세요.
    - `make install`
    - `make run_trigger`
- 'final-project-level3-recsys-13/modeling/core 폴더에 db관련 키 파일 넣기
    - .env : db연결을 위한 key
    - storage.json : storage 연결을 위한 key
- 'final-project-level3-recsys-13/modeling/core 폴더에 storage 관련 json파일 넣기
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
    - recbole...
- argparse 사용법(subparse)
    - argparse는 모든 모델에서 사용하는 공동 argparse와 하위 각 모델에서 사용할 subparse로 구성됨
    - 예시1) als를 사용하고 싶을 때
        - `python train.py als`
    - 예시2) bert2vec에서 공동 argparse인 'seed'와 bert2vec subparse인 'top_n'를 사용하고 싶을 때
        - `python train.py --seed 777 bert2vec --top_n 30`
- train에서는 모델 학습 및 wandb 로그 저장만 진행되며, inference는 따로 구현되어 있음.

### Inference  
- **inference 하기 전**
    - 다음 쉘 커맨드로 batch tag 업데이트(batch tag는 최근 업데이트된 모델을 구별하주는 역할을 합니다.)
    - `python inference.py --update_batch_tag`
- inference : 커맨드는 train과 같음
    - infernce.py는 모델을 모든 데이터로 학습시키고, 결과를 google storage로 전송시킵니다.
    - storage 모델 저장: {모델이름}.pickle
        - ex) BPR.pickle

- **inference 한 후**
    - 다음 쉘 커맨드로 최근 batch tag를 가진 모델 중 가장 높은 recall을 가진 모델들의 이름을 db에 저장합니다. 
    - `python inference.py --inference_info`

