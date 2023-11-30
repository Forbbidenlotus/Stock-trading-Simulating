
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json
from matplotlib.dates import DateFormatter
import requests
import csv
import numpy as np


def reverse_csv_rows(input_file, output_file):
    # read the input CSV file
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    # reverse the entire row excluding the header (first row)
    header = rows[0]
    reversed_rows = [header] + rows[1:][::-1]

    # write the modified rows to the output CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(reversed_rows)


def download_csv(ticker, FILE_PATH):
    CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol='+ticker+'&interval=5min&slice=year1month1&apikey=L5VUO9GPIRFS30EN'
    with requests.Session() as s:
        download = s.get(CSV_URL)
        with open(FILE_PATH, 'wb') as f:
            f.write(download.content)
            print(f"File saved to {FILE_PATH}")


def fetch_time_series_data(file_path):
    return file_path


def trading_simulation(file_path, num_shares=1, buy_threshold=-2, sell_threshold=3):#changing the threshold change when to buy or sell
    # Load data from CSV file
    data = pd.read_csv(file_path, usecols=['time', 'open'])

    previous_price = None
    buy_price = None
    sell_price = None
    profit = 0.0
    loss = 0.0
    timestamps = []
    profits = []

    for i, row in data.iterrows():
        timestamp = row['time']
        price_data = row['open']

        if previous_price is not None:
            price_diff = price_data - previous_price
            percent_diff = (price_diff / previous_price) * 100

            # Buy shares if buy_price is not set and percent_diff is above buy_threshold
            if buy_price is None and percent_diff >= buy_threshold:
                buy_price = price_data
                print(f"Buying {num_shares} share(s) at {buy_price:.2f} ({timestamp})")

            # Sell shares if sell_price is not set and percent_diff is below sell_threshold or it's the end of the day
            elif buy_price is not None and (percent_diff <= sell_threshold or i == len(data) - 1):
                sell_price = price_data
                trade_profit = (sell_price - buy_price) * num_shares

                if trade_profit >= 0:
                    profit += trade_profit
                    print(f"Selling {num_shares} share(s) at {sell_price:.2f} ({timestamp}), profit: {trade_profit:.2f}")
                else:
                    loss += abs(trade_profit)
                    print(f"Selling {num_shares} share(s) at {sell_price:.2f} ({timestamp}), loss: {-trade_profit:.2f}")

                buy_price = None


        previous_price = price_data
        timestamps.append(timestamp)
        profits.append(profit - loss)
    print(f"Sum of Profit: {profit:.2f}, Sum of Loss: {loss:.2f}")
    print(f"Total profit: {profit-loss:.2f}")

    return timestamps, profits






def plot_profit_and_loss(timestamps, profits):
    plt.figure(figsize=(15, 6))

    # Convert timestamp strings to datetime objects
    datetime_timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in timestamps]


    plt.plot(datetime_timestamps, profits, label='Profit & Loss')
    plt.xlabel('Timestamp')
    plt.ylabel('Profit ($)')
    plt.title('Profit & Loss Throughout the Time')
    plt.grid()
    plt.legend()

    # Format the x-axis dates
    date_format = DateFormatter("%Y-%m-%d %H:%M")

    plt.gca().xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)

    plt.show()
def get_historical_prices(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = json.loads(response.text)
    if 'Time Series (Daily)' not in data:
        raise ValueError('Invalid API response')
    return data['Time Series (Daily)']

def calculate_volatility(prices):
    daily_returns = []
    for date, data in prices.items():
        daily_returns.append(float(data['5. adjusted close']) / float(data['4. close']) - 1)
    volatility = np.std(daily_returns) * np.sqrt(252)
    return volatility


def main():
    ticker = input("Enter ticker symbol: ")

    api_key = 'L5VUO9GPIRFS30EN'

    prices = get_historical_prices(ticker, api_key)
    volatility = calculate_volatility(prices)

    print(f"The volatility of {ticker} is {volatility:.2f}")



    download_csv(ticker, f"{ticker}.csv")
    reverse_csv_rows(f"{ticker}.csv", f"{ticker}_mod.csv")
    print(f"Simulating trading for {ticker} using data from the CSV file")
    data = fetch_time_series_data(f"{ticker}_mod.csv")
    timestamps, profits = trading_simulation(data)

    plot_profit_and_loss(timestamps, profits)

if __name__ == "__main__":
    main()