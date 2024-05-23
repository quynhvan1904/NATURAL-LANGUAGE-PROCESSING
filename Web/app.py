from flask import Flask, render_template, request
import os
from joblib import load
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
import numpy as np
import string
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')

app = Flask(__name__)

# models_folder = r'C:\Users\KIM\OneDrive - ueh.edu.vn\UEH\UEH\HK 5\Natural Language Processing\Final Project\Code test\train_model'
models_folder = os.path.dirname(os.path.abspath(__file__))

w2v_model_path = os.path.join(models_folder, 'model_w2v')
w2v_model = Word2Vec.load(w2v_model_path)
lstm_model_topic_path = os.path.join(models_folder, 'model_LSTM_topic.h5')
lstm_model_topic = load_model(lstm_model_topic_path)
lstm_model_sentiment_path = os.path.join(models_folder, 'model_LSTM_sentiment.h5')
lstm_model_sentiment = load_model(lstm_model_sentiment_path)
# Load Random Forest models for topic and sentiment
classifier_topic_path = os.path.join(models_folder, 'model_random_forest_topic.joblib')
classifier_topic = load(classifier_topic_path)
classifier_sentiment_path = os.path.join(models_folder, 'model_random_forest_sentiment.joblib')
classifier_sentiment = load(classifier_sentiment_path)
# Load Logistic Regression models for topic and sentiment
logistic_model_topic_path = os.path.join(models_folder, 'model_Logistic_Regression_topic.joblib')
logistic_model_topic = load(logistic_model_topic_path)
logistic_model_sentiment_path = os.path.join(models_folder, 'model_Logistic_Regression_sentiment.joblib')
logistic_model_sentiment = load(logistic_model_sentiment_path)

punctuations=list(string.punctuation)
def normalize_text(s):
    s = re.sub(r'\b(colon\w+)\b', " ", s)
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r'\d', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()

    tokens = []
    for sent in sent_tokenize(s):
        words = word_tokenize(sent)
        tokens.extend(words)
    return tokens

def get_vector(word_list, model):
    # Khởi tạo một vector 0
    vec = np.zeros(model.vector_size).reshape((1, model.vector_size))
    count = 0.
    for word in word_list:
        # Thêm vector của từ vào vec
        vec += model.wv.get_vector(word).reshape((1, model.vector_size))
        count += 1.
    if count != 0:
        vec /= count
    return vec

def preprocess_and_embed(text, w2v_model):
        tokenized_text = normalize_text(text)
        vector = get_vector(tokenized_text, w2v_model)
        vector = vector.reshape(-1, 1, w2v_model.vector_size)
        return tokenized_text, vector

def predict_topic(sentence, w2v_model, lstm_model=None, rf_model=None, lr_model=None):
        tokenized_text, embedding = preprocess_and_embed(sentence, w2v_model)

        if lstm_model:
            prediction = lstm_model.predict(embedding)
            topic_labels = ['lecturer', 'training_program', 'facility', 'others']
            predicted_topic = topic_labels[np.argmax(prediction)]
            return predicted_topic

        elif rf_model:
            prediction = rf_model.predict([embedding.flatten()])
            topic_labels = ['lecturer', 'training_program', 'facility', 'others']
            predicted_topic = topic_labels[prediction[0]]
            return predicted_topic

        elif lr_model:
            prediction = lr_model.predict([embedding.flatten()])
            topic_labels = ['lecturer', 'training_program', 'facility', 'others']
            predicted_topic = topic_labels[prediction[0]]
            return predicted_topic

def predict_sentiment(sentence, w2v_model, lstm_model=None, rf_model=None, lr_model=None):
        tokenized_text, embedding = preprocess_and_embed(sentence, w2v_model)

        if lstm_model:
            prediction = lstm_model.predict(embedding)
            sentiment_labels = ['negative', 'positive']
            predicted_sentiment = sentiment_labels[np.argmax(prediction)]
            return predicted_sentiment

        elif rf_model:
            prediction = rf_model.predict([embedding.flatten()])
            sentiment_labels = ['negative', 'positive']
            predicted_sentiment = sentiment_labels[prediction[0]]
            return predicted_sentiment

        elif lr_model:
            prediction = lr_model.predict([embedding.flatten()])
            sentiment_labels = ['negative', 'positive']
            predicted_sentiment = sentiment_labels[prediction[0]]
            return predicted_sentiment

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text_to_predict = request.form['text_to_predict']
        
        # Predict Topic
        predicted_topic_lr = predict_topic(text_to_predict, w2v_model, lr_model=logistic_model_topic)
        predicted_topic_lstm = predict_topic(text_to_predict, w2v_model, lstm_model=lstm_model_topic)
        predicted_topic_rf = predict_topic(text_to_predict, w2v_model, rf_model=classifier_topic)

        # Predict Sentiment
        predicted_sentiment_rf = predict_sentiment(text_to_predict, w2v_model, rf_model=classifier_sentiment)
        predicted_sentiment_lr = predict_sentiment(text_to_predict, w2v_model, lr_model=logistic_model_sentiment)
        predicted_sentiment_lstm = predict_sentiment(text_to_predict, w2v_model, lstm_model=lstm_model_sentiment)

        return render_template('result.html',
                               text_to_predict=text_to_predict,
                               predicted_topic_rf=predicted_topic_rf,
                               predicted_topic_lr=predicted_topic_lr,
                               predicted_topic_lstm=predicted_topic_lstm,
                               predicted_sentiment_rf=predicted_sentiment_rf,
                               predicted_sentiment_lr=predicted_sentiment_lr,
                               predicted_sentiment_lstm=predicted_sentiment_lstm)

if __name__ == '__main__':
    app.run(debug=True)
