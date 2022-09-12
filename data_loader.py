import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import seaborn as sns
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from collections import defaultdict

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

def preprocess(sentence):
    
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) in range(3,14) if not w in stopwords.words('english')]

    # return None is sentence have only one word
    if " ".join(filtered_words).count(" ") == 0: return
    return " ".join(filtered_words)

def stemming(sentence):
    
    stem_sentence = [ps.stem(word) for word in sentence.split()]
    return ' '.join(stem_sentence)

def count_top_x_words(corpus, top_x, skip_top_n):
    count = defaultdict(lambda: 0)
    for c in corpus:
        for w in word_tokenize(c):
            count[w] += 1
    count_tuples = sorted([(w, c) for w, c in count.items()], key=lambda x: x[1], reverse=True)
    return [i[0] for i in count_tuples[skip_top_n: skip_top_n + top_x]]


def replace_top_x_words_with_vectors(corpus, top_x):
    topx_dict = {top_x[i]: i for i in range(len(top_x))}

    return [
        [topx_dict[w] for w in word_tokenize(s) if w in topx_dict]
        for s in corpus
    ], topx_dict


def filter_to_top_x(corpus, n_top, skip_n_top=0):
    top_x = count_top_x_words(corpus, n_top, skip_n_top)
    return replace_top_x_words_with_vectors(corpus, top_x)

df = pd.read_csv('data.csv')
# Standarize columns names
df = df.rename(columns={'clean_text': 'text', 'is_depression':'label'})

df['clean_text'] = df['text'].map(lambda s:preprocess(s)) 
df = df.dropna()

ps = PorterStemmer()
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