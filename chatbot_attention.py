from tensorflow.keras.layers import Input, LSTM, Dense, Dot
from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import Embedding, TimeDistributed
from tensorflow.keras.models import Model, save_model
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import pickle5 as pickle
import pandas as pd
import os

os.chdir("c:/Users/soohan/project2020")

# 단어 목록 dict를 읽어온다.
with open('./dataset/vocabulary.pickle', 'rb') as f:
    word2idx,  idx2word = pickle.load(f)
    
# 학습 데이터 : 인코딩, 디코딩 입력, 디코딩 출력을 읽어온다.
with open('./dataset/train_data.pickle', 'rb') as f:
    trainXE, trainXD, trainYD = pickle.load(f)
	
# 평가 데이터 : 인코딩, 디코딩 입력, 디코딩 출력을 만든다.
with open('./dataset/eval_data.pickle', 'rb') as f:
    testXE, testXD, testYD = pickle.load(f)
    
VOCAB_SIZE = len(idx2word)
EMB_SIZE = 128
LSTM_HIDDEN = 128
MODEL_PATH = './dataset/Attention.h5'
LOAD_MODEL = False


def Attention(x, y):
    score = Dot(axes=(2, 2))([y, x])
    
    dist = Activation('softmax')(score)
    
    attention = Dot(axes=(2, 1))([dist, x])
    
    return Concatenate()([y, attention])  

K.clear_session()
wordEmbedding = Embedding(input_dim=VOCAB_SIZE, output_dim=EMB_SIZE)

# Encoder
encoderX = Input(batch_shape=(None, trainXE.shape[1]))
encEMB = wordEmbedding(encoderX)
encLSTM1 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state = True)
encLSTM2 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state = True)
ey1, eh1, ec1 = encLSTM1(encEMB)    # LSTM 1층 
ey2, eh2, ec2 = encLSTM2(ey1)       # LSTM 2층

# Decoder
decoderX = Input(batch_shape=(None, trainXD.shape[1]))
decEMB = wordEmbedding(decoderX)
decLSTM1 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
decLSTM2 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
dy1, _, _ = decLSTM1(decEMB, initial_state = [eh1, ec1])
dy2, _, _ = decLSTM2(dy1, initial_state = [eh2, ec2])
att_dy2 = Attention(ey2, dy2)
decOutput = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))
outputY = decOutput(att_dy2)

# Model
model = Model([encoderX, decoderX], outputY)
model.compile(optimizer=optimizers.Adam(lr=0.001), 
              loss='sparse_categorical_crossentropy')

if LOAD_MODEL:
    model.load_weights(MODEL_PATH)

# 학습 (teacher forcing)
hist = model.fit([trainXE, trainXD], trainYD, batch_size = 300, 
                 epochs=500, shuffle=True,
                 validation_data = ([testXE, testXD], testYD))

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label = 'Test loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 학습 결과 저장
save_model(model_enc, './dataset/model_enc')
save_model(model_dec, './dataset/model_dec')
save_model(model, MODEL_PATH)
     
    
