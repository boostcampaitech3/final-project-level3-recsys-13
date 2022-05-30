# final_project_modeling_base
![](https://velog.velcdn.com/images/whattsup_kim/post/54274827-c482-4418-929e-93d2e863bc22/image.png)

## How To Use
### setting
- (OpenBLAS Warning이 나오는 경우)
    - 쉘에 `export OPENBLAS_NUM_THREADS=1`
- CUDA 설치
    - `conda config --append channels conda-forge`
    - `conda install cudatoolkit=11.2 -c conda-forge`
    - `conda install cudatoolkit-dev=11.2 -c conda-forge`
- .env에 db정보 작성
    - ./.env

### 데이터 불러오기 및 train/test set 분리 (혹은 db데이터가 업데이트 되었다면, 실행해야 함)
- train set, test set, answer 업데이트 (1분 정도 소요)
    - `python data_split.py`
    - 데이터가 업데이트 되었으므로, tag도 1이 늘어납니다.
- 정답과 비슷한 레시피 정보 업데이트 (1시간 정도 소요)
    - `python similar_answer.py`

### Train  
- 모델 학습
    - `python train.py`
- 학습 가능한 모델
    - 모든 모델 학습: `--model all`
    - als: `--model als`
    - ....
> - modeling/experiments.csv에 학습한 모델과 eval 점수, 날짜, batch-serving-tag 등을 저장합니다. (batch-serving-tag: batch serving을 위한 학습 시 +1이 되는 int 타입의 tag(tag가 높아질수록 최신 모델임을 의미합니다.))

### Inference  
- modeling/inference.py에서 inference()를 호출하여 사용
    - input: 모델 이름(str), 사용자 id(int), 추천받을 개수(int)
    - output: 추천 아이템(list)

## TODO(모델링 Part)
1. inference
    - 함수 input에 모델을 명시하지 않으면, modeling/experiments.csv에 저장된 (가장 높은 tag를 가진)모델들 중 eval 결과가 가장 높은 모델로 inference합니다.  
    (modeling/experiments.csv 기반 자동 inference)
1. db연결
    - 필요하다면, (코드 - train, eval 폴더에 있는 데이터) 모든 모델의 데이터 로더 및 args 수정해야 함.
2. airflow 
    - 데이터 업데이트 시
        - `python data_split.py`
        - `python similar_answer.py`
    - 배치 단위 학습
        - `python train.py --model all`
3. wandb or mlflow 연결
    - 실험 추적을 보다 용이하게 하기 위한 협업 툴 설정
    - modeling/experiments.csv 파일은 유지? (고려해야 할 부분)
        - wandb 혹은 mlflow에서 좋은 성능을 가진 모델을 뽑아낼 수 있는가?
        - wandb 혹은 mlflow에서 모델을 패키징하여 배포할 수 있는가?
    - 더 나은 inference를 위한 고민
4. 많은 모델 실험