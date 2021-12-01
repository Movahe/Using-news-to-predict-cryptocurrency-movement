import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import csv

def pre_process_price(price_df):
    """
    """
    price_df.rename(columns={'Date':'date', 'Open':'open', 'Close':'close'}, inplace=True)
    price_df['date'] = price_df['date'].apply(lambda x: datetime.datetime.strptime(x, '%d-%b-%y'))
    price_df['open'] = price_df['open'].apply(lambda x: float(x.replace(',', '')))
    price_df['close'] = price_df['close'].apply(lambda x: float(x.replace(',', '')))
    price_df['High'] = price_df['High'].apply(lambda x: float(x.replace(',', '')))
    price_df['Low'] = price_df['Low'].apply(lambda x: float(x.replace(',', '')))
    price_df['Volume'] = price_df['Volume'].apply(lambda x: float(x.replace(',', '')) if x != '-' else np.nan)
    price_df['Volume'] = price_df['Volume'].fillna(price_df['Volume'].mean())
    price_df['Market Cap'] = price_df['Market Cap'].apply(lambda x: float(x.replace(',', '')))
    #res_df = price_df[['date', 'open', 'close']]
    return price_df

def pre_process_news(news_df):
    """
    Read from cryptonews.csv
    """
    news_df['date'] = news_df['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    
    return news_df

def combine_news(curr_df, crypto_df):
    """
    combine news_data set together
        args:
            curr_df(DataFrame): the processed dataset from cryptocurrency.news
            crypto_df(DataFrame): the processed dataset from cryptonews.com
        returns:
            df(DataFrame): the combined dataframe of curr_df and crypto_df
    """
    frames = [curr_df, crypto_df]
    newsOfBitcoin = pd.concat(frames, ignore_index=True)
    return newsOfBitcoin

def add_label(price_df):
    """
    """
    price_df['diff'] = price_df['close'] - price_df['open']
    price_df['label'] = [1 if x > 0 
                         else 0 for x in price_df['diff']]
    return price_df