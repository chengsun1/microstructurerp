#%% data preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def load_data():
    data = pd.read_csv('LDN_60s.csv')
    data['dateFormatted'] = pd.to_datetime(data['DateTime'].str[:26],
                                           format='%Y-%m-%dT%H:%M:%S.%f')
    data['mElasped']=data.dateFormatted.dt.hour*60+data.dateFormatted.dt.minute

    return data

def filter_data(data):
    data = data[(data.mElasped > 60*8+14) & (data.mElasped < 16.5*60-4)].copy()
    data.drop(columns=['DateTime','mElasped'], inplace = True)

    return data

num_stocks = 3
def select_stocks(data, num_stocks):
    np.random.seed(0)
    unique_stocks = data['Stock'].unique()
    myStock = np.random.choice(unique_stocks, num_stocks, replace=False)
    data = data[data['Stock'].isin(myStock)]

    return data

def add_columns(data):
    data['mid'] = 0.5*(data['Bid']+data['Ask'])
    data['spread'] = 10000*(data['Ask']-data['Bid'])/data['mid']
    data['depth'] = 0.5*(data['AskSize']+data['BidSize'])

    return data

data = load_data()
data = filter_data(data)
data = select_stocks(data,num_stocks)
data = add_columns(data)
data = data.mask(data.spread <= 0)
#%%
# 1.1 Plot the mid-price time series for each stock
uniqueStock = data['Stock'].unique()
for stock in uniqueStock:
    subset = data[data['Stock'] == stock]
    plt.figure()
    plt.plot(subset['dateFormatted'], subset['mid'])
    plt.title(str(stock) + ' Mids')
    plt.xlabel('Time')
    plt.ylabel('Mid Price')
    plt.savefig(f'{str(stock)}_mid.png')
    plt.close()


