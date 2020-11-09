from tensorflow.keras.layers import Input, LSTM, Dense, Dot
from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import Embedding, TimeDistributed
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np
import pickle5 as pickle
import tensorflow

import sys
sys.getdefaultencoding() 

# 단어 목록 dict를 읽어온다.
with open('./dataset/vocabulary.pickle', 'rb') as f:
    word2idx,  idx2word = pickle.load(f)
    
VOCAB_SIZE = len(idx2word)
EMB_SIZE = 128
LSTM_HIDDEN = 128
MAX_SEQUENCE_LEN = 10            # 단어 시퀀스 길이
MODEL_PATH = './dataset/Attention.h5'

def Attention(x, y):
    score = Dot(axes=(2, 2))([y, x])   

    dist = Activation('softmax')(score)               

    attention = Dot(axes=(2, 1))([dist, x])
    
    return Concatenate()([y, attention])   

# 워드 임베딩 레이어. Encoder와 decoder에서 공동으로 사용한다.
K.clear_session()
wordEmbedding = Embedding(input_dim=VOCAB_SIZE, output_dim=EMB_SIZE)

# Encoder
encoderX = Input(batch_shape=(None, MAX_SEQUENCE_LEN))
encEMB = wordEmbedding(encoderX)
encLSTM1 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state = True)
encLSTM2 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state = True)
ey1, eh1, ec1 = encLSTM1(encEMB)    # LSTM 1층 
ey2, eh2, ec2 = encLSTM2(ey1)       # LSTM 2층

# Decoder
decoderX = Input(batch_shape=(None, 1))
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
model.load_weights(MODEL_PATH)

# Chatting용 model
model_enc = Model(encoderX, [eh1, ec1, eh2, ec2, ey2])

ih1 = Input(batch_shape = (None, LSTM_HIDDEN))
ic1 = Input(batch_shape = (None, LSTM_HIDDEN))
ih2 = Input(batch_shape = (None, LSTM_HIDDEN))
ic2 = Input(batch_shape = (None, LSTM_HIDDEN))
ey = Input(batch_shape = (None, MAX_SEQUENCE_LEN, LSTM_HIDDEN))

dec_output1, dh1, dc1 = decLSTM1(decEMB, initial_state = [ih1, ic1])
dec_output2, dh2, dc2 = decLSTM2(dec_output1, initial_state = [ih2, ic2])
dec_attention = Attention(ey, dec_output2)
dec_output = decOutput(dec_attention)
model_dec = Model([decoderX, ih1, ic1, ih2, ic2, ey], 
                  [dec_output, dh1, dc1, dh2, dc2])

# Question을 입력받아 Answer를 생성한다.
def genAnswer(question):
    question = question[np.newaxis, :]
    init_h1, init_c1, init_h2, init_c2, enc_y = model_enc.predict(question)

    word = np.array(word2idx['<START>']).reshape(1, 1)

    answer = []
    for i in range(MAX_SEQUENCE_LEN):
        dY, next_h1, next_c1, next_h2, next_c2 = \
            model_dec.predict([word, init_h1, init_c1, init_h2, init_c2, enc_y])
        
        nextWord = np.argmax(dY[0, 0])
        
        if nextWord == word2idx['<END>'] or nextWord == word2idx['<PADDING>']:
            break
        
        answer.append(idx2word[nextWord])
        
        word = np.array(nextWord).reshape(1,1)
    
        init_h1 = next_h1
        init_c1 = next_c1
        init_h2 = next_h2
        init_c2 = next_c2
        
    return ' '.join(answer)

# Chatting
def chatting(n=100):
    for i in range(n):
        question = input('Q : ')
        
        if  question == 'quit':
            break
        
        q_idx = []
        for x in question.split(' '):
            if x in word2idx:
                q_idx.append(word2idx[x])
            else:
                q_idx.append(word2idx['<UNKNOWN>'])
        

        if len(q_idx) < MAX_SEQUENCE_LEN:
            q_idx.extend([word2idx['<PADDING>']] * (MAX_SEQUENCE_LEN - len(q_idx)))
        else:
            q_idx = q_idx[0:MAX_SEQUENCE_LEN]
        
        answer = genAnswer(np.array(q_idx))
        print('A :', answer)

chatting(100)
