import numpy as np
from django.shortcuts import render
from tensorflow.keras.models import load_model
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
import pickle

model = load_model('./savedModels/my_model')

number_of_words = 3440
standard_len = 155
ps = PorterStemmer()

a_file = open("data.pkl", "rb")
word_list = pickle.load(a_file)

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

def model_prediciton(model, sentence):
    
    sentence = preprocess(sentence)
    sentence = stemming(sentence)
    vectorize_sentence = []
    for word in sentence.split():
        if word in word_list:
            vectorize_sentence.append(word_list[word])
    
    while len(vectorize_sentence) < standard_len:vectorize_sentence.insert(0,0) 
        
    # Normalizing sentence
    vectorize_sentence = np.reshape(np.asarray(vectorize_sentence) / number_of_words, (1,155))
    
    prediciton = model.predict(vectorize_sentence)
    
    return prediciton

def predictor(request):
    if request.method == 'POST':
        pred_sentence = request.POST['pred_sentence']
        y_pred = model_prediciton(model=model, sentence=pred_sentence)
        if y_pred[0][0] > .5: y_pred = 'Sentence indicate depression with {}% probability'.format(np.round(100*y_pred[0][0], 2))
        else:y_pred = 'Probability of depression is only {}%'.format(np.round(100*y_pred[0][0], 2))
        return render(request, 'main.html', {'result' : y_pred})
    return render(request, 'main.html')
