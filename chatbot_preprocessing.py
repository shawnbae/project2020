import json
import pandas as pd

# 데이터셋 받아오기
with open("C:/Users/soohan/Downloads/감성대화말뭉치.json", "rb") as json_file:
    original_data= json.load(json_file)

# 데이터 가공하기
question_list= list()
for data in original_data:
  for questions in pd.Series(data['talk']['content'])[['HS01','HS02','HS03']]: 
    question_list.append(questions)
    
answer_list= list()
for data in original_data:
  for answers in pd.Series(data['talk']['content'])[['SS01','SS02','SS03']]: 
    answer_list.append(answers)
    
qa_dictionary= {'question': question_list, 'answer': answer_list}
dataset= pd.DataFrame(qa_dictionary)
dataset.to_csv('C:/Users/soohan/dataset.csv')


