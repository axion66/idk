### EEG뇌파실험

## 공개할 목적이 아니였던지라 파일들이 상당히 더러울 수 있습니다...

### 1 실험:
    뇌에 16개의 전극을 붙이고, 10개의 단어를 구분할 수 있는지 실험.
    3초동안 10개의 다른 단어중 하나를 보여주었고, 각각 단어들 사이엔 3초의 텀.(classification문제)
    모든 전극에서의 초반 3초는 제거함(3초 공백으로 시작함; dataset_maker가 알아서 제거함.)
    2 실험:
    똑같이 16개전극, 허나 3개분류(red,green,blue)
    초반 4초는 공백, 그 후 2초 단어, 2초 공백 무한반복
### 목적: 
    10개의 다른 단어들을 뇌파를 통해 구분

### mne library:
    뇌파 라이브러리로 쉽게 BDF파일을 np.array로 변환해주거나 ICA전처리를 해줌 


### 모델관련:
    model conclusion: 10개의 다중분류라 찍어도 10%인데 정확도가 9.7~13%사이로 나옴 
        -> 쉬운 모델들로 훈련가능한지 연구했으나 아무것도 배우지않음
        -> K-fold cross validation 이랑 train/val/test는 하지않음(검증할 정확도도 안나옴). train/test로만 했음.
        심지어 rgb는 훈련조차 안됨(모델이 각기다른 input에 대해 같은 output(mean)을 출력. -> 아마 특징을 찾기 어려워서일것임.

    loss_fn = nn.CrossEntropyLoss(모델 마지막에 softmax를 넣지않았음)
    optimizer = nn.Adam
    batch_size = 8,16,32전부 사용
    lr = 3e-4,1e-4,3e-5전부 사용
    epoch = 3,300,1000,100000전부 사용


    orig data는 Z-score normalization을 적용한 orig_data, min-max normalization을 적용한 orig_data, 그냥 orig_data중 하나. 

    orig data:
        orig data:
            LSTM(2LSTM + FC),
            CNNLSTM(3CNN + 2LSTM + FC),
            attn(I used it, but I don't think it was needed as simple LSTM or CNNLSTM didn't work(those had accuracy of ~11%.))
            SVM & K-mean(scikit-learn껄 썼으나 잘 쓸줄 몰라서인지 결과가 좋지는 않았음)

        orig data -> spectrogram:스팩트로그램이 노이즈의 학습방해를 줄인다 하여 사용해봄
            LSTM(2LSTM + FC),
            CNNLSTM(3CNN + 2LSTM + FC),
            attn(I used it, but I don't think it was needed as simple LSTM or CNNLSTM didn't work(cuz those had accuracy of ~11%.))

    


### 데이터와 라벨의 형식:
    test5부터test17가 데이터. test1부터4는 실수로 라벨을 랜덤으로 설정..(포함안함)
    the number "15" in "test5_15.BDF" means it contains 15 min of recording. (sample rate:100)
    만약 "test7.BDF"같이 시간을 안나타낸다면 10분을 의미. 

    모든 10분은 같은 텍스트 라벨을 가짐
    모든 15분도 같은 텍스트 라벨을 가짐
    1개의 30분은 자기 혼자이기에 혼자 라벨을 가짐

    1분에는 총 10개의 단어가 있음(3초 단어1, 3초 휴식, 3초단어2, 3초휴식 ...)-> 모두 dataset_maker가 처리할것임
