import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
from data_loader import standard_len, number_of_words, df, mapped_list, pred_sentence
from ML_model import my_model, model_prediciton

tf.random.set_seed(
                seed=0
)

labels = np.asarray(df['label'].tolist()).astype('float64')

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

if __name__ == '__main__':

    history = model.fit(x_train, y_train, epochs=20, batch_size=20)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test Loss: {}, Test Accuracy: {}'.format(test_loss, test_acc))

    y_pred = model.predict(x_test)
    y_class = np.round(y_pred)

    pd.DataFrame(history.history).plot(figsize=(8,5))

    confusion_mtx = confusion_matrix(y_test, y_class)
    fig, ax = plt.subplots(figsize=(12,8))
    ax = heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel('Prediciton Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    prediction = model_prediciton(model=model, sentence=pred_sentence)

    if prediction[0][0] > .5:print('Sentence indicate depression with {}% probability'.format(np.round(100*prediction[0][0], 2)))
    else:print('Probability of depression is only {}%'.format(np.round(100*prediction[0][0], 2)))

    plt.show()