# -*- coding: utf-8 -*-
"""NLP | Source code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15Z5jS-Of3eMimWO2KkxI1czXx0tLLsEe

link github: https://github.com/kimvo646/NLP.git

# Cài đặt

## pip install
"""

"""## Thư viện"""

# Xử lý dữ liệu
import pandas as pd
import numpy as np
import string
import re
import nltk
from underthesea import sent_tokenize, word_tokenize
from itertools import chain
from collections import Counter
from joblib import load

# Trực quan hóa
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from wordcloud import WordCloud

# Mô hình
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

## LSTM
import gensim
from gensim.models import Word2Vec
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from joblib import dump

# Khác
import warnings
warnings.filterwarnings("ignore")

"""## input"""

link_train = 'https://raw.githubusercontent.com/kimvo646/NLP/main/Web_demo/Data_csv/train_data.csv'
train = pd.read_csv(link_train, sep='\t', encoding='utf-16')
train.head(10)

link_test = 'https://raw.githubusercontent.com/kimvo646/NLP/main/Web_demo/Data_csv/test_data.csv'
test = pd.read_csv(link_test, sep='\t', encoding='utf-16')
test.head()

link_valid = 'https://raw.githubusercontent.com/kimvo646/NLP/main/Web_demo/Data_csv/validation_data.csv'
valid = pd.read_csv(link_valid, sep='\t', encoding='utf-16')
valid.head()

"""# Tổng quan bộ dữ liệu"""

# Tổng số dòng, số cột của bộ dữ liệu
print ('Các cột hiện có của bộ dữ liệu:')
for x in train.columns:
  print('>', x)
print(f"Bộ dữ liệu bao gồm {train.shape[1]} cột và {train.shape[0]} dòng")

"""# Tiền xử lý

## Kiểm tra các giá trị bị thiếu
"""

train.info()

train.isna().sum()

"""## Tách từ"""

def normalize_text(s):
    # Uncased
    s = s.lower()

    # Remove punctuations
    s = ''.join(ch for ch in s if ch not in string.punctuation)

    # Remove entities
    s = re.sub(r'\b((\w+|)wzjwz\d+)\b', " ", s)

    # Remove numbers
    s = re.sub(r'\d', ' ', s)

     # Fix whitespaces
    s = re.sub(r'\s+', ' ', s)

    #Remove leading and trailing spaces
    s = s.strip()

    return s

def tokenizer(text):
    tokens = []
    for sent in sent_tokenize(text):
        words = word_tokenize(sent)
        tokens.extend(words)
    return tokens

"""##Text mining"""

train_x=train['sentence'].tolist()
train_x=[normalize_text(sentence)for sentence in train_x]
all_tokens_train = [tokenizer(sentence) for sentence in train_x]

valid_x = valid['sentence'].tolist()
valid_x=[normalize_text(sentence)for sentence in valid_x]
all_tokens_valid = [tokenizer(sentence) for sentence in valid_x]

test_x = test['sentence'].tolist()
test_x=[normalize_text(sentence)for sentence in test_x]
all_tokens_test = [tokenizer(sentence) for sentence in test_x]


"""## Word2Vec
Nguồn tham khảo W2v: https://github.com/namlv97/biLSTM-vietnamese-uit-student-feedbacks/blob/main/biLSTM_Vietnamese_uit_student_feedbacks.ipynb
"""

# Cài đặt các chỉ số
min_count=1
window=3
vector_size=300
alpha=1e-3
min_alpha=1e-4
negative=10

word_sents_train=[sent for sent in all_tokens_train]
# Tạo mô hình Word2Vec
w2v_model = Word2Vec(min_count=min_count, window=window, vector_size=vector_size, alpha=alpha, min_alpha=min_alpha, negative=negative, sg=1)
# Xây dựng từ điển cho tập dữ liệu
w2v_model.build_vocab(word_sents_train)
#Huấn luyện mô hình
w2v_model.train(word_sents_train, total_examples=w2v_model.corpus_count, epochs=100, report_delay=1,compute_loss=True)

# Huấn luyện cho tập all_tokens_test
word_sents_test = [sent for sent in all_tokens_test]
w2v_model.build_vocab(word_sents_test, update=True)
w2v_model.train(word_sents_test, total_examples=w2v_model.corpus_count, epochs=100, report_delay=1, compute_loss=True)

# Huấn luyện cho tập all_tokens_valid
word_sents_valid = [sent for sent in all_tokens_valid]
w2v_model.build_vocab(word_sents_valid, update=True)
w2v_model.train(word_sents_valid, total_examples=w2v_model.corpus_count, epochs=100, report_delay=1, compute_loss=True)

# Lưu mô hình sau khi huấn luyện
w2v_model.save("model_w2v")


"""# Huấn luyện mô hình

##Random Forest
"""

def sentence_to_vec(sentence, model):
    vecs = [model.wv[word] for word in sentence if word in model.wv]
    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(model.vector_size)

"""###Topic"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

train_y=train['topic'].tolist()
test_y = test['topic'].tolist()

# Tạo ma trận đặc trưng cho tập huấn luyện và kiểm tra
X_train_w2v = np.array([sentence_to_vec(sent, w2v_model) for sent in word_sents_train])
X_test_w2v = np.array([sentence_to_vec(sent, w2v_model) for sent in word_sents_test])

# Tạo mô hình Decision Tree
classifier_topic = RandomForestClassifier(random_state=42)

# Huấn luyện mô hình với dữ liệu huấn luyện Word2Vec
classifier_topic.fit(X_train_w2v, train_y)

# Lưu mô hình
dump(classifier_topic, 'model_random_forest_topic.joblib')

# Dự đoán với dữ liệu kiểm tra
predicted_y = classifier_topic.predict(X_test_w2v)

# Tính toán các chỉ số đánh giá
accuracy = accuracy_score(test_y, predicted_y)
recall = recall_score(test_y, predicted_y, average='weighted')
precision = precision_score(test_y, predicted_y, average='weighted')
f1 = f1_score(test_y, predicted_y, average='weighted')

# In ra các chỉ số
print(f'Accuracy: {accuracy}')
print(f'Recall:   {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1}')



"""###Sentiment"""

train_y=train['sentiment'].tolist()
test_y = test['sentiment'].tolist()

# Tạo ma trận đặc trưng cho tập huấn luyện và kiểm tra
X_train_w2v = np.array([sentence_to_vec(sent, w2v_model) for sent in word_sents_train])
X_test_w2v = np.array([sentence_to_vec(sent, w2v_model) for sent in word_sents_test])

# Tạo mô hình Random Forest
classifier_sentiment = RandomForestClassifier(random_state=42)

# Huấn luyện mô hình với dữ liệu huấn luyện Word2Vec
classifier_sentiment.fit(X_train_w2v, train_y)

# Lưu mô hình
dump(classifier_sentiment, 'model_random_forest_sentiment.joblib')

# Dự đoán với dữ liệu kiểm tra
predicted_y = classifier_sentiment.predict(X_test_w2v)

# Tính toán các chỉ số đánh giá
accuracy = accuracy_score(test_y, predicted_y)
recall = recall_score(test_y, predicted_y, average='weighted')
precision = precision_score(test_y, predicted_y, average='weighted')
f1 = f1_score(test_y, predicted_y, average='weighted')

# In ra các chỉ số
print(f'Accuracy: {accuracy}')
print(f'Recall:   {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1}')

