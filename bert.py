import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics import mean_squared_error, r2_score
# nltk.download('punkt')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    if pd.isna(text):  # 检查是否为NaN
        return ''  # 返回空字符串或其他默认值
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)
# 1. 加载数据
df = pd.read_csv('Review_train.csv')
df['review_time'] =  pd.to_datetime(df['review_time'])
df['review_time'] = df['review_time'].astype('int64')
# 2. 特征提取：将评论文本转换为 TF-IDF 特征
vectorizer = TfidfVectorizer(max_features=500)  # 限制最多提取 500 个特征
df['processed_text'] = df['text'].apply(preprocess_text)
X_text = vectorizer.fit_transform(df['processed_text']).toarray()

# 3. 添加额外特征（评论长度）
df['text_length'] = df['processed_text'].apply(len)
X = np.hstack((X_text, df[['text_length']].values))
X = np.hstack((X,df[['rating']].values,df[['review_time']].values))

# 4. 目标变量
y = df['log_helpful_vote'].values

# 5. 数据预处理：划分数据集并标准化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. 构建神经网络模型
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')  # 回归任务输出连续值
])

# 7. 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# 8. 训练模型
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8, verbose=1)

# 9. 测试模型
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_test - y_pred.flatten())**2))
mae = np.mean(np.abs(y_test - y_pred.flatten()))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f'R^2 Score: {r2}')

# 预测新评论的投票数量
new_reviews = ["Great product! Very useful.", "Poor quality, not worth it."]
new_X = vectorizer.transform(new_reviews).toarray()
new_X = np.hstack((new_X, np.array([len(r) for r in new_reviews]).reshape(-1, 1),))
new_X = scaler.transform(new_X)
predictions = model.predict(new_X)
print("Predicted useful votes:", predictions.flatten())