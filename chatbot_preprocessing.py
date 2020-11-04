import os
from konlpy.tag import Okt
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import pickle5 as pickle

os.chdir("C:/Users/soohan/project2020")

DATA_PATH = './dataset/recycle - recycle.csv'
TOKENIZE_AS_MORPH = False       # 형태소 분석 여부
ENC_INPUT = 0                   # encoder 입력을 의미함
DEC_INPUT = 1                   # decoder 입력을 의미함
DEC_TARGET = 2                  # decoder 출력을 의미함
MAX_SEQUENCE_LEN = 10           # 단어 시퀀스 길이

FILTERS = "([~.,!?\"':;)(])"
PAD = "<PADDING>"
STD = "<START>"
END = "<END>"
UNK = "<UNKNOWN>"

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)

# 판다스를 통해서 데이터를 불러와 학습 셋과 평가 셋으로 나누어 그 값을 리턴한다.
def load_data():
    data_df = pd.read_csv(DATA_PATH, header=0, sep=',')
    question, answer = list(data_df['question']), list(data_df['answer'])
    
    train_input, eval_input, train_label, eval_label = \
        train_test_split(question, answer, test_size=0.1, random_state=42)
        
    return train_input, train_label, eval_input, eval_label

def prepro_like_morphlized(data):
    morph_analyzer = Okt()
    result_data = list()
    for seq in tqdm(data):
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphlized_seq)

    return result_data

def data_processing(value, dictionary, pType):
    # 형태소 토크나이징 사용 유무
    if TOKENIZE_AS_MORPH:
        value = prepro_like_morphlized(value)

    sequences_input_index = []
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        
        if pType == DEC_INPUT:
            # 디코더 입력은 <START>로 시작한다.
            sequence_index = [dictionary[STD]]
        else:
            sequence_index = []
        
        for word in sequence.split():
            # word가 딕셔너리에 없으면 UNK (out of vacabulary)를 넣는다.
            if dictionary.get(word) is not None:
                sequence_index.append(dictionary[word])
            else:
                sequence_index.append(dictionary[UNK])
        
            # 문장의 단어수를  제한한다.
            if len(sequence_index) >= MAX_SEQUENCE_LEN:
                break
        
        # 디코더 출력은 <END>로 끝난다.
        if pType == DEC_TARGET:
            if len(sequence_index) < MAX_SEQUENCE_LEN:
                sequence_index.append(dictionary[END])
            else:
                sequence_index[len(sequence_index)-1] = dictionary[END]
                
        # max_sequence_length보다 문장 길이가 작으면 빈 부분에 PAD(0)를 넣어준다.
        sequence_index += (MAX_SEQUENCE_LEN - len(sequence_index)) * [dictionary[PAD]]
        sequences_input_index.append(sequence_index)

    return np.asarray(sequences_input_index)
  
# 토크나이징
def data_tokenizer(data):
    words = []
    for sentence in data:
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split():
            words.append(word)
    return [word for word in words if word]

# 사전 파일을 만든다
def make_vocabulary():
    data_df = pd.read_csv(DATA_PATH, encoding='utf-8', sep=',')
    question, answer = list(data_df['question']), list(data_df['answer'])
    
    # 질문과 응답 문장의 단어를 형태소로 바꾼다
    if TOKENIZE_AS_MORPH:  
        question = prepro_like_morphlized(question)
        answer = prepro_like_morphlized(answer)
        
    data = []
    data.extend(question)
    data.extend(answer)
    words = data_tokenizer(data)
    words = list(set(words)) # fit_on_texts()라는 함수로 빈도가 높은 순서대로 뽑아낼 수도 있다.
    words[:0] = MARKER

    word2idx = {word: idx for idx, word in enumerate(words)}
    idx2word = {idx: word for idx, word in enumerate(words)}
    
    # 두가지 형태의 키와 값이 있는 형태를 리턴한다. 
    # (예) 단어: 인덱스 , 인덱스: 단어)
    return word2idx, idx2word
  
  
# 질문과 응답 문장 전체의 단어 목록 dict를 만든다.
word2idx, idx2word = make_vocabulary()

# 질문과 응답 문장을 학습 데이터와 시험 데이터로 분리한다.
train_input, train_label, eval_input, eval_label = load_data()

# 학습 데이터 : 인코딩, 디코딩 입력, 디코딩 출력을 만든다.
train_input_enc = data_processing(train_input, word2idx, ENC_INPUT)
train_input_dec = data_processing(train_label, word2idx, DEC_INPUT)
train_target_dec = data_processing(train_label, word2idx, DEC_TARGET)
	
# 평가 데이터 : 인코딩, 디코딩 입력, 디코딩 출력을 만든다.
eval_input_enc = data_processing(eval_input, word2idx, ENC_INPUT)
eval_input_dec = data_processing(eval_label, word2idx, DEC_INPUT)
eval_target_dec = data_processing(eval_label, word2idx, DEC_TARGET)

# 결과를 저장한다.
with open('./dataset/vocabulary.pickle', 'wb') as f:
    pickle.dump([word2idx, idx2word], f, pickle.HIGHEST_PROTOCOL)

with open('./dataset/train_data.pickle', 'wb') as f:
    pickle.dump([train_input_enc, train_input_dec, train_target_dec], f, pickle.HIGHEST_PROTOCOL)

with open('./dataset/eval_data.pickle', 'wb') as f:
    pickle.dump([eval_input_enc, eval_input_dec, eval_target_dec], f, pickle.HIGHEST_PROTOCOL)
