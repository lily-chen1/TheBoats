import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


def load_saved_model_and_tokenizer(model_path, tokenizer_path):
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def load_data():
    restaurants = pd.read_csv(os.path.join("data", "restaurants.csv"))
    restaurants = restaurants[["review", "sentiment"]]
    restaurants["sentiment"] = restaurants["sentiment"].replace(
        {0: "negative", 1: "positive"}
    )

    steam = pd.read_csv(os.path.join("data", "steam.csv"))
    steam = steam[["review", "sentiment"]]
    steam["sentiment"] = steam["sentiment"].replace({0: "negative", 1: "positive"})
    return restaurants, steam


def preprocess_data(data, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences(data["review"])
    padded_sequences = pad_sequences(
        sequences, maxlen=max_len, padding="post", truncating="post"
    )
    label_encoder = LabelEncoder()

    labels = label_encoder.fit_transform(data["sentiment"])
    return padded_sequences, labels


def main():
    model_path = "sentiment_model"
    tokenizer_path = "tokenizer.pickle"
    max_len = 100

    model, tokenizer = load_saved_model_and_tokenizer(model_path, tokenizer_path)

    restaurants, steam = load_data()
    X_test, y_test = preprocess_data(restaurants, tokenizer, max_len)

    predictions = (model.predict(X_test) >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy on restaurant reviews data (easy dataset): {accuracy:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test, predictions, target_names=["negative", "positive"]
        )
    )

    X_test, y_test = preprocess_data(steam, tokenizer, max_len)

    predictions = (model.predict(X_test) >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy on Steam reviews data (hard dataset): {accuracy:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test, predictions, target_names=["negative", "positive"]
        )
    )


if __name__ == "__main__":
    main()
