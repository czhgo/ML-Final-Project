import pandas as pd
df =pd.read_csv('result.csv')
df = df[['review_id','log_useful_vote']]
df.to_csv('final_result.csv',index=False)