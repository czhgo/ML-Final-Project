import pandas as pd
import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 下载停用词
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 分词、去除停用词、词干提取
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
    df['word_count'] =df['processed_text'].apply(len)
    df['avg_word_length'] = df['processed_text'].apply(
        lambda x: np.mean([len(word) for word in x]) if x else 0)
    df['vocabulary_size'] = df['processed_text'].apply(lambda x: len(set(x)))

    # Topic modeling using LDA
    vectorizer = CountVectorizer(max_features=5000)
    review_matrix = vectorizer.fit_transform(df['processed_text'])
    lda = LatentDirichletAllocation(n_components=5, random_state=42,verbose=1)
    lda.fit(review_matrix)

    # Assign topic distribution
    df['topic_distribution'] = lda.transform(review_matrix).tolist()
    return df
def demo_context_forest(df1,df2):
    train_df =context_analysis(df1)
    test_df = context_analysis(df2)
    X_train = np.concatenate([np.array(train_df['topic_distribution'].tolist()),
                        train_df[['word_count', 'avg_word_length', 'vocabulary_size','review_time','rating',]].to_numpy()], axis=1)
    y_train = train_df['log_helpful_vote']    # 划分数据集
    X_test = np.concatenate([np.array(test_df['topic_distribution'].tolist()),
                        test_df[['word_count', 'avg_word_length', 'vocabulary_size','review_time','rating',]].to_numpy()], axis=1)

    # 训练模型

    model = GradientBoostingRegressor(n_estimators=30,verbose=1)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    df2['log_useful_vote'] = y_pred
    df2.to_csv('result.csv')
    # 评估模型

if __name__ == '__main__':
    df1 = pd.read_csv('Review_train.csv')
    df1['review_time'] = pd.to_datetime(df1['review_time'])
    df1['review_time'] = df1['review_time'].astype('int64')
    df2 = pd.read_csv('Review_test.csv')
    df2['review_time'] = pd.to_datetime(df2['review_time'])
    df2['review_time'] = df2['review_time'].astype('int64')
    demo_context_forest(df1,df2)

