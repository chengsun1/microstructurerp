#%% data preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import statsmodels.api as sm
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
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=500)

    ax1.plot(subset['dateFormatted'], subset[columns_to_plot[0]], color=colors[0])
    ax1.set_title(f'{str(stock)+ " " + "Depth and Spread" }', fontproperties=font)
    ax1.set_xlabel('Date', fontproperties=font)
    ax1.set_ylabel(columns_to_plot[0], color=colors[0], fontproperties=font)
    ax1.tick_params(axis='y', labelcolor=colors[0], labelsize=8)

    ax2 = ax1.twinx()
    ax2.plot(subset['dateFormatted'], subset[columns_to_plot[1]], color=colors[1])
    ax2.set_ylabel(columns_to_plot[1], color=colors[1], fontproperties=font)
    ax2.tick_params(axis='y', labelcolor=colors[1], labelsize=8)

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
daily_mean = data.groupby(['Stock','Date'])[['depth','spread']].mean().reset_index()
data['mid_return'] = data.groupby('Stock')['mid'].pct_change()
data['abs_mid_return'] = data['mid_return'].abs()
daily_volatility = data.groupby(['Stock','Date'])['abs_mid_return'].mean().reset_index()
daily_stats = pd.merge(daily_mean,daily_volatility,on=['Stock','Date'])
for stock in stocks:
    fig, ax1 = plt.subplots(figsize=(10,5), dpi=500)
    stock_data = daily_stats[daily_stats['Stock'] == stock]
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Mean Depth', color=color)
    ax1.plot(stock_data['Date'], stock_data['depth'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Daily Mean Spread', color=color)
    ax2.plot(stock_data['Date'], stock_data['spread'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax1.twinx()
    color = 'tab:green'
    ax3.set_ylabel('Daily Midquote Volatility', color=color)
    ax3.plot(stock_data['Date'], stock_data['abs_mid_return'], color=color)
    ax3.tick_params(axis='y', labelcolor=color)
    plt.title(f'Daily Mean Depth, Spread and Midquote Volatility for {stock}')
    plt.tight_layout()
    plt.savefig(f'{foutput_path}/{str(stock) + "daily_mean_spread_volatility.png"}',dpi=500)
#%% 1.3.2
correlation_table = pd.DataFrame(columns=['Stock','Correlation'])
for stock in stocks:
    stock_data = daily_stats[daily_stats['Stock'] == stock]
    correlation = stock_data['spread'].corr(stock_data['depth'])
    newrow = pd.DataFrame({'Stock':stock,'Correlation':correlation},index=[0])
    correlation_table = pd.concat([correlation_table,newrow],ignore_index=True)
    correlation_table.to_csv(f'{doutput_path}/correlation.csv')
#%% 1.3.3 linear regression
daily_stats = daily_stats.dropna(subset=['Stock'])
stocks = daily_stats['Stock'].unique()
regression_results = pd.DataFrame(columns=['Stock', 'Dependent', 'Intercept', 'Slope', 'p-value','R-squared'])
for stock in stocks:
    stock_data = daily_stats[daily_stats['Stock'] == stock]

    for dependent in ['spread', 'depth']:
        X = stock_data['abs_mid_return']
        y = stock_data[dependent]
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        regression_results = pd.concat([regression_results,
                                        pd.DataFrame({'Stock': [stock],
                                                      'Dependent': [dependent],
                                                      'Intercept': [round(results.params.const,4)],
                                                      'Slope': ['{:.4e}'.format(results.params.abs_mid_return)],
                                                      'p-value': [round(results.pvalues.abs_mid_return,4)],
                                                      'R-squared': [round(results.rsquared,4)]})],
                                       ignore_index=True)

regression_results.to_csv(f'{doutput_path}/regression.csv')







