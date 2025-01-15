import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 加载数据
df = pd.read_csv('Review_train.csv')  # 假设评论数据存储在reviews.csv中
# 定义停用词
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
stop_words = set(stopwords.words('english'))

# 文本预处理函数
def preprocess_text(text):
    if pd.isna(text):  # 检查是否为NaN
        return ''  # 返回空字符串或其他默认值
    # 去除符号和数字
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 分词
    words = word_tokenize(text.lower())
    # 去除停用词
    words = [word for word in words if word not in stop_words]
    return words

# 应用预处理
df['processed_text'] = df['text'].apply(preprocess_text)
model = Word2Vec(sentences=df['processed_text'], vector_size=300, window=10, min_count=2, workers=4, sg=1)
# 假设已知的维度词
dimensions = ['quality', 'service']
# dimensions = {
#     'food_quality': ['taste', 'flavor', 'fresh'],
#     'service': ['service', 'staff', 'waiter']
# }

# 提取维度词
dimension_words = {dim: model.wv.most_similar(dim, topn=10) for dim in dimensions}
print("Dimension Word Lists:")
for dim, words in dimension_words.items():
    print(f"{dim}: {words}")
# 统计维度变量
def calculate_dimension_stats(text, dimension_words):
    stats = {dim: {'Attri_nums': 0, 'Attri_avglen': 0} for dim in dimensions}
    for word in text:
        for dim, words in dimension_words.items():
            if word in [w[0] for w in words]:
                stats[dim]['Attri_nums'] += 1
                stats[dim]['Attri_avglen'] += len(word)
    for dim in dimensions:
        if stats[dim]['Attri_nums'] > 0:
            stats[dim]['Attri_avglen'] /= stats[dim]['Attri_nums']
    return stats
# 情感分析函数
# 初始化情感分析器
sia = SentimentIntensityAnalyzer()

# 情感分析函数
def analyze_sentiment(text):
    text = ' '.join(text)  # 将列表转换为字符串
    scores = sia.polarity_scores(text)
    return scores

# 应用情感分析函数
df['sentiment'] = df['processed_text'].apply(analyze_sentiment)

# 应用统计函数
df['dimension_stats'] = df['processed_text'].apply(lambda x: calculate_dimension_stats(x, dimension_words))
print(df[['text', 'dimension_stats']].head())
df.to_csv('Review_train.csv',index=False)
