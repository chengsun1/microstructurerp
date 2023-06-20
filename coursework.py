#%% data preprocessing
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
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
#%% 1.1
# 1.1.1 Plot the mid-price time series for each stock
stocks = data['Stock'].unique()
columns_to_plot = ['depth','spread']
colors = ['blue','red']
foutput_path = '/Users/chengsun/Documents/GitHub/microstructurerp/Outputs/Graph'
font = FontProperties()
font.set_family('sans-serif')

for stock in stocks:
    subset = data[data['Stock'] == stock]
    fig, axs = plt.subplots(nrows=2, figsize=(10, 5), dpi=500)

    for ax, col, color in zip(axs, columns_to_plot,colors):
        ax.plot(subset['dateFormatted'], subset[col], color=color)
        ax.set_title(f'{str(stock) + " " + col.capitalize()}', fontproperties = font)
        ax.set_xlabel('Time', fontproperties = font)
        ax.set_ylabel(col, fontproperties = font)

    plt.tight_layout()
    plt.savefig(f'{foutput_path}/{str(stock) + "time_series.png"}',dpi=500)
    plt.close()

# 1.1.2 summary statistics
doutput_path = '/Users/chengsun/Documents/GitHub/microstructurerp/Outputs/Data'
summary = data.groupby('Stock')[['depth','spread']].describe()
summary.to_csv(f'{doutput_path}/summary.csv')

#%% 1.2
# 1.2.1
data['Hour'] = data.dateFormatted.dt.hour
columns_depth_spread = ['depth','spread']
hourly_means = {}
for column in columns_depth_spread:
    hourly_means[column] = data.groupby(['Stock','Hour'])[column].mean().reset_index()
#1.2.2
for stock in stocks:
    fig, ax1 = plt.subplots(figsize=(10,5), dpi=500)

    depth_df = hourly_means['depth'][hourly_means['depth']['Stock'] == stock]
    spread_df = hourly_means['spread'][hourly_means['spread']['Stock'] == stock]

    color = 'tab:blue'
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Hourly Mean Depth', color=color)
    ax1.plot(depth_df['Hour'], depth_df['depth'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Hourly Mean Spread', color=color)
    ax2.plot(spread_df['Hour'], spread_df['spread'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Hourly Mean Depth and Spread for {stock}')
    plt.savefig(f'{foutput_path}/{str(stock) + "hourly_mean_spread.png"}',dpi=500)

#%% 1.3
data['Date']=data.dateFormatted.dt.date
#%% 2.1
data2 = pandas.read_excel('smm_pf_data_2023.xlsx')



