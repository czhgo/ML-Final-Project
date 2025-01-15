import pandas as pd

df = pd.read_csv('Review_train.csv')
duplicated_rows = df[df.duplicated(subset=['user_id', 'product_id'], keep=False)]
print(1)