import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Define tickers
# -----------------------------
tickers = ['XLE','BZ', 'NG', 'BNO',
    'NEE', 'BEP', 'FSLR', 'GEV',
    'CWEN', 'RNW',
    'EFT', 'CLNE', 'EQT', 'SM',
    'CRGY', 'NGS']

benchmark_ticker = '^GSPC'  # S&P 500
start_date = '2010-01-01'
end_date = '2026-11-20'

all_tickers = tickers + [benchmark_ticker]

# -----------------------------
# 2. Download data
# -----------------------------
print("Downloading data... please wait")

data = yf.download(
    all_tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False
)

# Get Close prices
if 'Close' in data.columns:
    prices = data['Close']
else:
    prices = data

print("Download finished")
print(prices.head())

# -----------------------------
# 3. Clean data - common sample
# -----------------------------
# Drop tickers with no data at all
prices = prices.dropna(axis=1, how='all')

print("\nFirst available observation date:")
for ticker in prices.columns:
    first_date = prices[ticker].dropna().index.min()
    print(f"{ticker}: {first_date}")

# Common start = latest first date
common_start = max(
    prices[ticker].dropna().index.min()
    for ticker in prices.columns
)
print("\nCommon sample starts on:", common_start)

prices = prices.loc[common_start:]
prices = prices.dropna(axis=1, thresh=int(0.8 * len(prices)))
prices = prices.ffill()
prices = prices.dropna()

# -----------------------------
# 4. Log returns + Summary Table
# -----------------------------
log_returns = np.log(prices / prices.shift(1))
log_returns = log_returns.dropna()

print("\nLog returns shape:")
print(log_returns.shape)

summary = pd.DataFrame({
    'Mean': log_returns.mean(),
    'Std Dev': log_returns.std(),
    'Min': log_returns.min(),
    'Max': log_returns.max(),
    'Skewness': log_returns.skew(),
    'Kurtosis': log_returns.kurtosis()
})

print("\n--- Descriptive Statistics ---")
print(summary)
summary.to_csv('code1_summary.csv')
print("Summary saved as code1_summary.csv")

# -----------------------------
# 5. Plot - Normalized Prices
# -----------------------------
normalized = prices / prices.iloc[0] * 100

plt.figure(figsize=(14, 7))
for ticker in tickers:
    if ticker in normalized.columns:
        plt.plot(normalized.index, normalized[ticker], label=ticker, linewidth=0.9)

if benchmark_ticker in normalized.columns:
    plt.plot(normalized.index, normalized[benchmark_ticker], label='S&P 500', color='black', linewidth=2.2)

plt.title('Normalized Prices: Portfolio vs S&P 500 (Common Sample)')
plt.xlabel('Date')
plt.ylabel('Normalized Price (Start = 100)')
plt.legend(ncol=4, fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('code1_picture.png')
plt.show()

print("Done! Picture saved as code1_picture.png")
