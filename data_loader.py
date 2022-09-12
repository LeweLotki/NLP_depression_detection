import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from preprocessing import preprocess, stemming, filter_to_top_x

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

df = pd.read_csv('data.csv')
# Standarize columns names
df = df.rename(columns={'clean_text': 'text', 'is_depression':'label'})

df['clean_text'] = df['text'].map(lambda s:preprocess(s)) 
df = df.dropna()

df['stem_text'] = df['clean_text'].map(lambda s:stemming(s)) 

longest_tweet = 0
sentences_len = []
for sentence in df['stem_text']: 
    sentence_len = " ".join(sentence.split()).count(" ")
    sentences_len.append(sentence_len)
    if longest_tweet < sentence_len:
        longest_tweet = sentence_len

avg = np.average(np.asarray(sentences_len))
std = np.std(np.asarray(sentences_len))
standard_len = int(np.ceil(avg + 2*std))

number_of_words = 0
counts = df["stem_text"].str.findall(r"(\w+)").explode().value_counts()
for idx, freq in enumerate(counts):
    if freq < 5:
        number_of_words = idx
        break

text_list = df['stem_text'].tolist()
mapped_list, word_list = filter_to_top_x(text_list, number_of_words)

with open('sentence.txt', 'r') as file:
    pred_sentence = file.read().replace('\n', ' ')