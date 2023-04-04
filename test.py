import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('data/train.csv', header=None, nrows=6000)

max_features = 10000
max_len = 150

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text

train_df[2] = train_df[2].apply(lambda x: clean_text(x))

def shift_number(x):
    return x - 1

train_df[0] = train_df[0].apply(shift_number)

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_df[2].values)
X = tokenizer.texts_to_sequences(train_df[2].values)
X = pad_sequences(X, maxlen=max_len)

y = pd.get_dummies(train_df[0]).values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, batch_size=32, epochs=1, validation_data=(X_val, y_val))

test_df = pd.read_csv('data/test.csv', header=None, nrows=1000)
test_df[2] = test_df[2].apply(lambda x: clean_text(x))
test_df[0] = test_df[0].apply(shift_number)
test_sequences = tokenizer.texts_to_sequences(test_df[2].values)
test_sequences = pad_sequences(test_sequences, maxlen=max_len)

y_pred = model.predict(test_sequences)

y_pred = np.argmax(y_pred, axis=1)

test_accuracy = np.mean(y_pred == test_df[0].values)
print('Test accuracy:', test_accuracy)