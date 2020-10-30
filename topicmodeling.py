import pandas as pd
from top2vec import Top2Vec

June= pd.read_excel('./dataset/NewsResult_20200601-20200630.xlsx')
July= pd.read_excel('./dataset/NewsResult_20200701-20200731.xlsx')
August= pd.read_excel('./dataset/NewsResult_20200801-20200831.xlsx')
September= pd.read_excel('./dataset/NewsResult_20200901-20200930.xlsx')
October= pd.read_excel('./dataset/NewsResult_20201001-20201029.xlsx')

daylist_June= June[['일자','키워드']].groupby('일자').apply(lambda d: ",".join(d['키워드']))
daylist_July= July[['일자','키워드']].groupby('일자').apply(lambda d: ",".join(d['키워드']))
daylist_August= August[['일자','키워드']].groupby('일자').apply(lambda d: ",".join(d['키워드']))
daylist_September= September[['일자','키워드']].groupby('일자').apply(lambda d: ",".join(d['키워드']))
daylist_October= October[['일자','키워드']].groupby('일자').apply(lambda d: ",".join(d['키워드']))

model = Top2Vec(documents=daylist_June[20200601].split(','), speed="learn", workers=5)
topic_sizes, topic_nums = model.get_topic_sizes()
print(topic_sizes)
print(topic_nums)

model.get_topics()[0]

for topic in topic_nums[:10]:
    model.generate_topic_wordcloud(topic)




