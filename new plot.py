import plotly.plotly as ply
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime
# ply.tools.set_credentials_file(username='a6sylee', api_key='vgxkgikmS2hk1IeKdWTo')

df = pd.read_csv('bitcoin_price.csv')
df.head()



trace = go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])
data = [trace]
ply.iplot(data, filename='privacy-public', sharing='public', auto_open=True)


