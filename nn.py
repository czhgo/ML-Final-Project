import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    if pd.isna(text):  # 检查是否为NaN
        return ''  # 返回空字符串或其他默认值
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)


def context_analysis(df):
    df['processed_text'] = df['text'].apply(preprocess_text)
    # Extract features
    df['word_count'] = df['processed_text'].apply(len)
    df['avg_word_length'] = df['processed_text'].apply(
        lambda x: np.mean([len(word) for word in x]) if x else 0)
    df['vocabulary_size'] = df['processed_text'].apply(lambda x: len(set(x)))

    # Topic modeling using LDA
    vectorizer = CountVectorizer(max_features=5000)
    review_matrix = vectorizer.fit_transform(df['processed_text'])
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(review_matrix)

    # Assign topic distribution
    df['topic_distribution'] = lda.transform(review_matrix).tolist()
    return df,vectorizer
def demo_context_forest(df):
    train_df,vectorizer = context_analysis(df)
    X = np.concatenate([np.array(df['topic_distribution'].tolist()),
                        train_df[['word_count', 'avg_word_length', 'vocabulary_size']].to_numpy()], axis=1)
    y = train_df['log_helpful_vote']  # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 模型参数
    vocab_size = len(vectorizer.vocabulary_)
    embedding_dim = 100
    max_length = X_train.shape[1]

    # 搭建模型
    model = Sequential([
        Embedding(vocab_size + 1, embedding_dim, input_length=max_length),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')
    # 预测
    y_pred = model.predict(X_test)

    # 评估模型
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
if __name__ == '__main__':
    df = pd.read_csv('review_with_help.csv')
    df = df.head(5000)
    demo_context_forest(df)