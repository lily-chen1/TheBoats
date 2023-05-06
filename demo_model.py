import pickle
import tensorflow as tf
from keras.utils import pad_sequences


def load_saved_model_and_tokenizer(model_path, tokenizer_path):
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def predict_sentiment(model, tokenizer, text, max_len):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(
        sequences, maxlen=max_len, padding="post", truncating="post"
    )

    prediction = model.predict(padded_sequences)

    sentiment = "positive" if prediction >= 0.5 else "negative"
    return sentiment


def main():
    model_path = "sentiment_model"
    tokenizer_path = "tokenizer.pickle"
    max_len = 100

    model, tokenizer = load_saved_model_and_tokenizer(model_path, tokenizer_path)

    while True:
        user_input = input(
            "Enter the text you want to analyze the sentiment of. Type 'quit' to exit: "
        )

        if user_input.lower() == "quit":
            break

        sentiment = predict_sentiment(model, tokenizer, user_input, max_len)
        print(f"Predicted sentiment: {sentiment}")


if __name__ == "__main__":
    main()
