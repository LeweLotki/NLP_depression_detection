import numpy as np
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, InputLayer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from preprocessing import preprocess, stemming
from data_loader import word_list, standard_len, number_of_words

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
    vectorize_sentence = np.reshape(np.asarray(vectorize_sentence) / number_of_words, (1,155))
    
    prediciton = model.predict(vectorize_sentence)
    
    return prediciton
