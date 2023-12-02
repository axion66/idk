EEG뇌파실험

*공개할 목적이 아니였던지라 파일들이 상당히 더러울 수 있음.

실험:
    뇌에 16개의 전극을 붙이고, 10개의 단어를 구분할 수 있는지 실험해봄.
    3초동안 10개의 다른 단어중 하나를 보여주었고, 각각 단어들 사이엔 3초의 텀이 있음.(classification문제)
    모든 전극에서의 초반 3초는 제거함(공백으로 시작하기떄문; dataset_maker가 알아서 제거함.)

목적: 
    10개의 다른 단어들을 뇌파를 통해 구분

mne library:
    는 뇌파 라이브러리로 쉽게 .BDF파일을 np.array로 변환해주거나 ICA전처리 관련 라이브러리를 포함함 


모델관련:
    model conclusion: 10개의 다중분류라 찍어도 10%인데 정확도가 9.7~13%사이로 나옴 
        -> 쉬운 모델들로 훈련가능한지 연구했으나 아무것도 배우지않음
        -> K-fold cross validation 이랑 train/val/test는 하지않음(검증할 정확도도 안나옴). train/test로만 했음.


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

    


데이터와 라벨의 형식:
    I have test5~test17.#test1~4는 나의 실수로 라벨을 랜덤으로 설정해버림..
    the number "15" in "test5_15.BDF" means it contains 15 min of recording. (sample rate:100)
    만약 "test7.BDF"같이 시간을 안나태낸다면 10분을 의미함.

    모든 10분은 같은 텍스트 라벨을 가짐.
    모든 15분도 같은 텍스트 라벨을 가짐
    1개의 30분은 자기 혼자이기에 혼자 라벨을 가짐

    1분에는 총 10개의 단어가 있음(3초 단어1, 3초 휴식, 3초단어2, 3초휴식 ...)