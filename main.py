import tensorflow as tf
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
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, InputLayer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
from data_loader import preprocess, stemming, standard_len, number_of_words, df, mapped_list, word_list

tf.random.set_seed(
                seed=0
)

class my_model:
    
    def __init__(self, embedding=0, ker=0, ker_2=0, pool_size=0, strides=0, input_shape=0):
        
        self.embedding = embedding
        self.ker = ker
        self.ker_2 = ker_2
        self.pool_size = pool_size
        self.strides = strides
        self.input_shape = input_shape
        
        
    def cnn(self):

        model = Sequential()

        model.add(Embedding(number_of_words, self.embedding, input_length=self.input_shape))
        model.add(Conv1D(self.ker, self.ker_2))
#         model.add(MaxPooling1D(pool_size=self.pool_size, strides=self.strides, padding='valid'))                            

        model.add(Flatten())
            
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        opt = Adam(learning_rate=1e-2)
        model.compile(loss='BinaryCrossentropy', optimizer=opt, metrics=['AUC'])

        return model
    
    def ann(self):

        model = Sequential()

        model.add(InputLayer(input_shape=(self.input_shape,)))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        opt = Adam(learning_rate=1e-2)
        model.compile(loss='BinaryCrossentropy', optimizer=opt, metrics=['AUC'])

        return model

def model_prediciton(model, sentence):
    
    sentence = preprocess(sentence)
    sentence = stemming(sentence)
    vectorize_sentence = []
    for word in sentence.split():
        if word in word_list:
            vectorize_sentence.append(word_list[word])
    
    while len(vectorize_sentence) < standard_len:vectorize_sentence.insert(0,0) 
        
    # Normalizing sentence
#     for idx, word in enumerate(vectorize_sentence):vectorize_sentence[idx] = word / 3440
    vectorize_sentence = np.reshape(np.asarray(vectorize_sentence) / number_of_words, (1,155))
    
    prediciton = model.predict(vectorize_sentence)
    
    return prediciton

labels = np.asarray(df['label'].tolist()).astype('float64')

max_text_length = standard_len # based on the statistic deviation
mapped_list = sequence.pad_sequences(mapped_list, maxlen=standard_len)

# Normalizing by division
mapped_list = mapped_list / number_of_words

x_train, x_test, y_train, y_test = train_test_split(mapped_list, labels, test_size=0.3)

model = my_model(
                embedding=50, 
                ker=50, 
                ker_2 = 20,
                pool_size=2,
                strides=2,
                input_shape=standard_len
)
model = model.ann()
model.summary()

history = model.fit(x_train, y_train, epochs=20, batch_size=20)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Loss: {}, Test Accuracy: {}'.format(test_loss, test_acc))

y_pred = model.predict(x_test)
y_class = np.round(y_pred)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()

confusion_mtx = confusion_matrix(y_test, y_class)
fig, ax = plt.subplots(figsize=(12,8))
ax = heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap='Blues')
ax.set_xlabel('Prediciton Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')
plt.show()

my_sentence = 'Isolation, empty feeling, easily agitated, nothing is interesting anymore, you wanna sleep all the time to forget about living. Eating more than usual when youre not hungry or not at all and still no hunger. Depression comes in many forms but a constant feeling of emptiness stays usually'

prediction = model_prediciton(model=model, sentence=my_sentence)

if prediction[0][0] > .5:print('Sentence indicate depression with {}% probability'.format(np.round(100*prediction[0][0], 2)))
else:print('Probability of depression is only {}%'.format(np.round(100*prediction[0][0], 2)))