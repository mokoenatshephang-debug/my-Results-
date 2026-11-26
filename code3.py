print("\nSummary Statistics")
print(summary)
# -----------------------------
# Save datasets
# -----------------------------
prices.to_csv('prices.csv')
log_returns.to_csv('log_returns.csv')
print("\nFiles saved:")
print("prices.csv")
print("log_returns.csv")
# -----------------------------
# Plot cumulative returns
# -----------------------------
cum_returns = np.exp(log_returns.cumsum())

plt.figure(figsize=(14,7))

for ticker in cum_returns.columns:
    plt.plot(cum_returns.index,
             cum_returns[ticker],
             label=ticker)
plt.title('Cumulative Growth of $1 Investment')
plt.xlabel('Date')
plt.ylabel('Growth')
plt.legend(loc='best', ncol=3)
#plt.grid(True)
plt.show()
# -----------------------------
# Plot cumulative returns
# -----------------------------
plt.figure(figsize=(14,7))
for ticker in data_com_returns.columns:
    plt.plot(data_com_returns.index,
             data_com_returns[ticker],
             label=ticker)
plt.title('Cumulative Growth of $1 Investment')
plt.xlabel('Date')
plt.ylabel('Growth')
plt.legend(loc='best', ncol=3)
#plt.grid(True)
plt.show()
# -----------------------------
# Plot cumulative returns
# -----------------------------
plt.figure(figsize=(14,7))

for ticker in data_com_returns1.columns:
    plt.plot(data_com_returns1.index,
             data_com_returns1[ticker],
             label=ticker)
plt.title('Cumulative Growth of $1 Investment')
plt.xlabel('Date')
plt.ylabel('Growth')
plt.legend(loc='best', ncol=3)
#plt.grid(True)
plt.show()
# Select the 15 energy stocks
tickers = ['XLE',
    'BEP', 'BZ', 'BNO', 'CLNE', 'CRGY',
    'CWEN', 'EFT', 'EQT', 'FSLR', 'GEV',
    'NEE', 'NG', 'NGS', 'RNW', 'SM'
]

# Select stocks from existing cumulative returns
data_com_returns = data_com_returns[tickers]

# Select corresponding logarithmic returns
data_returns = log_returns[tickers]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Annualisation factor for daily data
trading_days = 252

# Risk-free rate
risk_free_rate = 0.04

# Descriptive statistics
mean_returns = data_returns.mean() * trading_days

volatility = data_returns.std() * np.sqrt(trading_days)

variance = data_returns.var() * trading_days

sharpe_ratio = (
    mean_returns - risk_free_rate
) / volatility

# Create summary table
performance = pd.DataFrame({
    'Mean Annual Return': mean_returns,
    'Annual Variance': variance,
    'Annual Volatility': volatility,
    'Sharpe Ratio': sharpe_ratio
})

display(performance.sort_values(
    by='Sharpe Ratio',
    ascending=False
).round(4))
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Annualised returns
mean_returns.sort_values().plot(
    kind='barh',
    ax=axes[0]
)

axes[0].set_title('Annualised Mean Returns')
axes[0].set_xlabel('Annual Return')

# Annualised volatility
volatility.sort_values().plot(
    kind='barh',
    ax=axes[1]
)

axes[1].set_title('Annualised Volatility')
axes[1].set_xlabel('Annual Volatility')

plt.tight_layout()
plt.show()
correlation_matrix = data_returns.corr()

plt.figure(figsize=(13, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    center=0
)

plt.title('Correlation Matrix of Energy Stocks')
plt.tight_layout()
plt.show()
from pypfopt import (
    expected_returns,
    risk_models,
    EfficientFrontier,
    objective_functions,
    BlackLittermanModel,
    black_litterman
)

# Convert logarithmic returns to simple returns
simple_returns = np.expm1(data_returns)

# Split observations into training and testing periods
split = int(len(simple_returns) * 0.8)

past_returns = simple_returns.iloc[:split]
future_returns = simple_returns.iloc[split:]

# Convert returns into price indices for PyPortfolioOpt
past_prices = (1 + past_returns).cumprod()

future_prices = (1 + future_returns).cumprod()

# Expected annual returns
mu = expected_returns.mean_historical_return(
    past_prices,
    frequency=252
)

# Annualised covariance matrix
S = risk_models.sample_cov(
    past_prices,
    frequency=252
)

print("Expected annual returns:")
display(mu.round(4))

print("Covariance matrix:")
display(S.round(4))
ef_sharpe = EfficientFrontier(mu, S)

weights_sharpe = ef_sharpe.max_sharpe(
    risk_free_rate=risk_free_rate
)

performance_sharpe = ef_sharpe.portfolio_performance(
    verbose=True,
    risk_free_rate=risk_free_rate
)

weights_sharpe = ef_sharpe.clean_weights()
ef_min = EfficientFrontier(mu, S)

weights_min = ef_min.min_volatility()

performance_min = ef_min.portfolio_performance(
    verbose=True,
    risk_free_rate=risk_free_rate
)

weights_min = ef_min.clean_weights()
weights_df = pd.DataFrame({
    'Maximum Sharpe': weights_sharpe,
    'Minimum Volatility': weights_min
})

display(weights_df.round(4))

weights_df.plot(
    kind='bar',
    figsize=(14, 6)
)

plt.title('Mean-Variance Portfolio Weights')
plt.xlabel('Stocks')
plt.ylabel('Portfolio Allocation')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
from pypfopt import expected_returns

# Risk-free rate
risk_free_rate = 0.04

# Market benchmark
market_returns = log_returns['XLE'].iloc[:split]

# Convert market log returns to simple returns
market_returns = np.expm1(market_returns)

# Construct market price index
market_prices = (1 + market_returns).cumprod()

# CAPM expected returns
capm_returns = expected_returns.capm_return(
    past_prices,
    market_prices=market_prices,
    risk_free_rate=risk_free_rate,
    frequency=252
)

print("CAPM Expected Annual Returns:")
display(capm_returns.sort_values(ascending=False).round(4))
plt.figure(figsize=(12, 6))

capm_returns.sort_values().plot(
    kind='barh'
)

plt.title('CAPM Expected Returns for Energy Stocks')
plt.xlabel('Expected Annual Return')
plt.ylabel('Stock')

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 6))

capm_returns.sort_values().plot(
    kind='barh'
)

plt.title('CAPM Expected Returns for Energy Stocks')
plt.xlabel('Expected Annual Return')
plt.ylabel('Stock')

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
# Market capitalisation dictionary
# Replace these values with actual market capitalisations
# obtained for the selected stocks.

market_caps = pd.Series(
    1.0,
    index=past_prices.columns
)

# Market-implied risk aversion
delta = black_litterman.market_implied_risk_aversion(
    past_prices,
    frequency=252
)

# Equilibrium returns
market_prior = black_litterman.market_implied_prior_returns(
    market_caps,
    delta,
    S,
    risk_free_rate=risk_free_rate
)
# CAPM estimates used as absolute views
views = capm_returns.to_dict()

bl = BlackLittermanModel(
    S,
    pi=market_prior,
    absolute_views=views,
    tau=0.05
)

# Posterior returns and covariance
bl_returns = bl.bl_returns()
bl_cov = bl.bl_cov()

# Optimisation
ef_bl = EfficientFrontier(
    bl_returns,
    bl_cov
)

weights_bl = ef_bl.max_sharpe(
    risk_free_rate=risk_free_rate
)

performance_bl = ef_bl.portfolio_performance(
    verbose=True,
    risk_free_rate=risk_free_rate
)

weights_bl = ef_bl.clean_weights()

display(pd.Series(weights_bl, name='Black-Litterman Weights'))
# Actual annualised returns during testing period
future_prices = (1 + future_returns).cumprod()

future_rets = expected_returns.mean_historical_return(
    future_prices,
    frequency=252
)

# Return estimation methods
return_methods = {
    "Mean Historical Returns": mu,

    "EMA Historical Returns":
        expected_returns.ema_historical_return(
            past_prices,
            frequency=252
        ),

    "CAPM Returns": capm_returns,

    "Black-Litterman Returns": bl_returns
}

# Calculate MAE
mean_abs_errors = {}

for method, estimated_returns in return_methods.items():

    actual = future_rets.reindex(
        estimated_returns.index
    )

    mae = np.mean(
        np.abs(estimated_returns - actual)
    )

    mean_abs_errors[method] = mae

# Results
mae_results = pd.Series(
    mean_abs_errors,
    name='Mean Absolute Error'
).sort_values()

display(mae_results.to_frame())
plt.figure(figsize=(11, 6))

plt.barh(
    mae_results.index,
    mae_results.values
)

plt.xlabel('Mean Absolute Error')
plt.ylabel('Expected Return Estimation Method')

plt.title(
    'Comparison of Expected Return Estimation Methods'
)

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
portfolio_weights = {
    'Mean-Variance Max Sharpe': weights_sharpe,
    'Mean-Variance Min Volatility': weights_min,
    'Black-Litterman': weights_bl
}

portfolio_test_returns = {}

for name, weights in portfolio_weights.items():

    w = pd.Series(weights).reindex(
        future_returns.columns
    ).fillna(0)

    w = w / w.sum()

    portfolio_test_returns[name] = future_returns @ w

portfolio_test_returns = pd.DataFrame(
    portfolio_test_returns
)

# Cumulative portfolio growth
cumulative_portfolio = np.exp(
    np.log1p(portfolio_test_returns).cumsum()
)

cumulative_portfolio.plot(
    figsize=(14, 7),
    linewidth=2
)

plt.title('Out-of-Sample Portfolio Performance')
plt.xlabel('Date')
plt.ylabel('Growth of $1 Investment')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
