import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pickle


def load_data():
    data = []

    # load the 5 csv files we're using, preprocess them, and append them to the data list
    # this was a bit weird to do since the data format is different for each csv file but the idea is to just
    # have them end up as one dataframe that has two columns, the text and whether it is negative or positive
    # some csv files had sentiment columns that included things like neutral sentiment and these were just filtered out

    amazon = pd.read_csv(os.path.join("data", "amazon.csv"))
    amazon = amazon[["sentiment", "review"]]
    amazon["sentiment"] = amazon["sentiment"].replace({1: "negative", 2: "positive"})
    data.append(amazon)

    imdb = pd.read_csv(os.path.join("data", "imdb.csv"))
    imdb = imdb[["review", "sentiment"]]
    imdb["sentiment"] = imdb["sentiment"].replace({0: "negative", 1: "positive"})
    data.append(imdb)

    twitter = pd.read_csv(os.path.join("data", "twitter.csv"))
    twitter = twitter[["sentiment", "tweet"]]
    twitter = twitter[twitter["sentiment"].isin(["Positive", "Negative"])]
    twitter["sentiment"] = twitter["sentiment"].str.lower()
    twitter.columns = ["sentiment", "review"]
    data.append(twitter)

    financial = pd.read_csv(os.path.join("data", "financial.csv"))
    financial = financial[["sentence", "sentiment"]]
    financial = financial[financial["sentiment"].isin(["positive", "negative"])]
    financial.columns = ["review", "sentiment"]
    data.append(financial)

    restaurants = pd.read_csv(os.path.join("data", "restaurants.csv"))
    restaurants = restaurants[["review", "sentiment"]]
    restaurants["sentiment"] = restaurants["sentiment"].replace(
        {0: "negative", 1: "positive"}
    )
    data.append(restaurants)

    # turns the data list into a pandas dataframe and drops the rows that have missing columns
    all_data = pd.concat(data, axis=0, ignore_index=True)
    all_data = all_data.dropna(subset=["review"])

    print(all_data)
    return all_data


def preprocess_data(data, max_len):
    tokenizer = Tokenizer()

    # fit_on_texts creates a dictionary and assigns each word as a index in the dictionary
    tokenizer.fit_on_texts(data["review"])

    # texts_to_sequences replaces each word in the text data with the index from the dictionary to create a sequence
    sequences = tokenizer.texts_to_sequences(data["review"])

    # we need to make sure that each sequence has the same length so pad_sequences adds the
    # 0 index to each sequence until each sequence has the same length as the longest sequence
    padded_sequences = pad_sequences(
        sequences, maxlen=max_len, padding="post", truncating="post"
    )

    label_encoder = LabelEncoder()

    # turn the positive and negative values in the sentiment column into 0 and 1 labels
    labels = label_encoder.fit_transform(data["sentiment"])

    return padded_sequences, labels, tokenizer, label_encoder


def create_model(vocab_size, max_len, embedding_dim):
    # the Sequential class lets us create a linear stack of layers
    # the Embedding layer lets us turn our sequence of dictionary indexes into vectors
    # the Bidirectional layer lets us create a recurrent neural network with the LSTM layer
    # the LSTM layer implements a long short-term memory network. I honestly can't really explain what a LSTM is but I know that it's good for NLP apparently
    # GlobalMaxPool1D downsamples the inputs
    # the Dense layer applies an activation function to the inputs, which in this case is ReLU, which will prevent negative inputs
    # the Droupout layer is meant to prevent overfitting, and it randomly sets input units to 0
    # finally we have our last layer with is a Dense layer that applies the sigmoid activation function to give us our final label predictions

    model = Sequential(
        [
            Embedding(vocab_size, embedding_dim, input_length=max_len),
            Bidirectional(LSTM(64, return_sequences=True)),
            GlobalMaxPool1D(),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    # we use the Adam optimizer to apply a stochastic gradient descent method and we use a binary crossentropy loss function since we are doing binary classification
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def main():
    data = load_data()
    print(data["sentiment"].value_counts())
    max_len = 100
    test_size = 0.2
    batch_size = 64
    epochs = 5

    padded_sequences, labels, tokenizer, _ = preprocess_data(data, max_len)

    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, labels, test_size=test_size, random_state=42
    )

    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 128

    model = create_model(vocab_size, max_len, embedding_dim)
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
    )

    model.save("sentiment_model")
    with open("tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f)


if __name__ == "__main__":
    main()
