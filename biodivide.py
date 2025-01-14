import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(review_matrix)

    # Assign topic distribution
    df['topic_distribution'] = lda.transform(review_matrix).tolist()
    return df
def demo_context_forest(df):
    train_df =context_analysis(df)
    X = np.concatenate([np.array(train_df['topic_distribution'].tolist()),train_df[['word_count','avg_word_length','vocabulary_size']].to_numpy()],axis=1)
    y = train_df['anyvote']    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # 生成分类报告
    print(classification_report(y_test, y_pred))

    # 生成混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', conf_matrix)
if __name__ == '__main__':
    df = pd.read_csv('Review_train.csv')
    df['anyvote'] = df['helpful_vote']>0
    df = df.head(5000)
    demo_context_forest(df)
