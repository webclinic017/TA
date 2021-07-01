# Imports
import base64
import streamlit as st
from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si
from pandas import ExcelWriter
import yfinance as yf
import pandas as pd
import time
import os
import datetime
import plotly.graph_objects as go
import pandas_ta as ta


yf.pdr_override()



my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1)

# Variables
st.sidebar.header('User Input Features')
index = st.sidebar.multiselect('Index', ['S&P 500', 'Dow 30', 'Nasdaq', 'NYSE Composite Index'])
if not index:
    st.warning('Please choose an index.')
    st.stop()
    st.success('Thank you for choosing an index.')
ind = str(index)[2:-2]
if ind == 'Dow 30':
    selected_index = "^DJI",
    selected_ticker = si.tickers_dow()
elif ind == 'Nasdaq':
    selected_index = "^IXIC",
    selected_ticker = si.tickers_nasdaq()
elif ind == 'NYSE Composite Index':
    selected_index = "^NYA",
    selected_ticker = si.tickers_nifty50()
else:
    st.sidebar.header(index)
    selected_index = "^IXIC",
    selected_ticker = si.tickers_nasdaq()
tickers = selected_ticker
tickers = [item.replace(".", "-") for item in tickers]  # Yahoo Finance uses dashes instead of dots
index_name = selected_index  # S&P 500
start_date = datetime.datetime.now() - datetime.timedelta(days=365)
end_date = datetime.date.today()
exportList = pd.DataFrame(
    columns=['Stock', "RS_Rating", "50 Day MA", "150 Day Ma", "200 Day MA", "52 Week Low", "52 week High", "EMA12",
             "EMA26", "EMA Over"])
returns_multiples = []

# Index Returns
index_df = pdr.get_data_yahoo(index_name, start_date, end_date)
index_df['Percent Change'] = index_df['Adj Close'].pct_change()
index_return = (index_df['Percent Change'] + 1).cumprod()[-1]

# Find top 30% performing stocks (relative to the S&P 500)
stocks_num = st.sidebar.number_input("Number of Stocks", min_value=1, step=1, max_value=len(tickers))

for ticker in tickers[:stocks_num]:
    # Download historical data as CSV for each stock (makes the process faster)
    df = pdr.get_data_yahoo(ticker, start_date, end_date)
    outname = f'{ticker}.csv'
    outdir = f'files/{ind}/'
    if os.path.exists(outdir):
        pass
    else:
        os.mkdir(outdir)
    fullname = os.path.join(outdir, outname)
    df.to_csv(fullname)

    # Calculating returns relative to the market (returns multiple)
    df['Percent Change'] = df['Adj Close'].pct_change()
    stock_return = (df['Percent Change'] + 1).cumprod()[-1]

    returns_multiple = round((stock_return / index_return), 2)
    returns_multiples.extend([returns_multiple])
    print(f'Ticker: {ticker}; Returns Multiple against {ind}: {returns_multiple}\n')
    time.sleep(1)

# Creating dataframe of only top 30%
rs_df = pd.DataFrame(list(zip(tickers, returns_multiples)), columns=['Ticker', 'Returns_multiple'])
rs_df['RS_Rating'] = rs_df.Returns_multiple.rank(pct=True) * 100
rs_df = rs_df[rs_df.RS_Rating >= rs_df.RS_Rating.quantile(.70)]

# Checking Minervini conditions of top 30% of stocks in given list
rs_stocks = rs_df['Ticker']
for stock in rs_stocks:
    try:
        df = pd.read_csv(f'files/{ind}/{stock}.csv', index_col=0)
        sma = [50, 150, 200]
        for x in sma:
            df["SMA_" + str(x)] = round(df['Adj Close'].rolling(window=x).mean(), 2)
        # Storing required values
        currentClose = df["Adj Close"][-1]
        moving_average_50 = df["SMA_50"][-1]
        moving_average_150 = df["SMA_150"][-1]
        moving_average_200 = df["SMA_200"][-1]
        low_of_52week = round(min(df["Low"][-260:]), 2)
        high_of_52week = round(max(df["High"][-260:]), 2)
        RS_Rating = round(rs_df[rs_df['Ticker'] == stock].RS_Rating.tolist()[0])
        ema12 = ta.ema(ta.ohlc4(df["Open"], df["High"], df["Low"], df["Close"]), length=12)
        ema26 = ta.ema(ta.ohlc4(df["Open"], df["High"], df["Low"], df["Close"]), length=26)
        if ema12[-1] > ema26[-1]:
            ema_over = "Over"
        else:
            ema_over = "Stand"

        try:
            moving_average_200_20 = df["SMA_200"][-20]



        except Exception:
            moving_average_200_20 = 0

        # Condition 1: Current Price > 150 SMA and > 200 SMA
        condition_1 = currentClose > moving_average_150 > moving_average_200

        # Condition 2: 150 SMA and > 200 SMA
        condition_2 = moving_average_150 > moving_average_200

        # Condition 3: 200 SMA trending up for at least 1 month
        condition_3 = moving_average_200 > moving_average_200_20

        # Condition 4: 50 SMA> 150 SMA and 50 SMA> 200 SMA
        condition_4 = moving_average_50 > moving_average_150 > moving_average_200

        # Condition 5: Current Price > 50 SMA
        condition_5 = currentClose > moving_average_50

        # Condition 6: Current Price is at least 30% above 52 week low
        condition_6 = currentClose >= (1.3 * low_of_52week)

        # Condition 7: Current Price is within 25% of 52 week high
        condition_7 = currentClose >= (.75 * high_of_52week)

        # If all conditions above are true, add stock to exportList
        if (
                condition_1):
            # and condition_2 and condition_3 and condition_4 and condition_5 and condition_6 and condition_7
            exportList = exportList.append({'Stock': stock, "RS_Rating": RS_Rating, "50 Day MA": moving_average_50,
                                            "150 Day Ma": moving_average_150, "200 Day MA": moving_average_200,
                                            "52 Week Low": low_of_52week, "52 week High": high_of_52week,
                                            "EMA12": ema12[-1], "EMA26": ema26[-1], "EMA Over": ema_over},
                                           ignore_index=True)

            print(stock + " made the Minervini requirements")
    except Exception as e:
        print(e)
        print(f"Could not gather data on {stock}")
st.title(str(ind), ' App')

st.markdown("""
This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
""")

# apply doesn't work with selection
# def change_colour(val):
#     return ['background-color: red' if x == "Over" else 'background-color: green' for x in val]
# exportList = exportList.style.apply(change_colour, axis=1, subset=['EMA Over'])

exportList = exportList.sort_values(by='RS_Rating', ascending=False)

st.dataframe(exportList)
writer = ExcelWriter("ScreenOutput.xlsx")
exportList.to_excel(writer, "Sheet1")
writer.save()

# Sidebar - Sector selection
sorted_sector_unique = sorted(exportList['Stock'].unique())
selected_sector = st.sidebar.multiselect('Stock', sorted_sector_unique, sorted_sector_unique)


# Filtering data
df_selected_sector = exportList[(exportList['Stock'].isin(selected_sector))]

st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(
    df_selected_sector.shape[1]) + ' columns.')
st.dataframe(df_selected_sector)


def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="{ind}.csv">Download CSV File</a>'
    return href


st.markdown(filedownload(exportList), unsafe_allow_html=True)

# https://pypi.org/project/yfinance/

data = yf.download(
    tickers=list(exportList.Stock),
    period="ytd",
    interval="1d",
    group_by='ticker',
    auto_adjust=True,
    prepost=True,
    threads=True,
    proxy=None
)


# Plot Closing Price of Query Symbol
# def price_plot(symbol):
#     # df = pd.DataFrame(exportList['Stock']['Close'])
#     df = pdr.get_data_yahoo(symbol, start_date, end_date)
#     df['Date'] = df.index
#     plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
#     plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
#     plt.xticks(rotation=90)
#     plt.title(symbol, fontweight='bold')
#     plt.xlabel('Date', fontweight='bold')
#     plt.ylabel('Closing Price', fontweight='bold')
#     return st.pyplot()

def price_plot(symbol):
    df = pdr.get_data_yahoo(symbol, start_date, end_date)
    df['Date'] = df.index
    # plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
    # df.iplot(kind="candle",keys=["Open", "High", "Low", "Close"])
    # fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']])
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                         open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'])])
    fig.update_layout(
        title={
            'text': symbol,
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    return fig.show()


# num_company = st.sidebar.slider('Number of Companies', 1, 5)
with st.sidebar.form(key='Stock  Charts'):
    stock_charts = st.multiselect('Stock  Charts', sorted_sector_unique)
    submit_button = st.form_submit_button(label='Submit')


if st.button('Show Plots'):
    st.header('Stock Closing Price')
    for i in stock_charts:
        price_plot(i)
