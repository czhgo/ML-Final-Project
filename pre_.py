import pandas as pd
import numpy as np

def bio_classify_helpful(df):
    review_data_without_help = df[df['helpful_vote']==0]
    review_data_with_help = df[df['helpful_vote']!=0]
    return review_data_without_help,review_data_with_help

if __name__=='__main__':
    filename = 'Review_train.csv'
    df = pd.read_csv(filename)
    df = df.iloc[:, 2:]
    print(df.shape)
    df1,df2=bio_classify_helpful(df)
    print(df1.shape)
    print(df2.shape)
    df['anyvote']=df['helpful_vote']>0
    df['anyvote']=df['anyvote'].astype(int)
    df.to_csv('Review_train.csv')
    df2.to_csv('review_with_help.csv')