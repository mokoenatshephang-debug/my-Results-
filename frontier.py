import subprocess, sys
try:
    import yfinance as yf, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from scipy.optimize import minimize
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "numpy", "pandas", "matplotlib", "scipy"])
    import yfinance as yf, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from scipy.optimize import minimize

tickers = ['XLE','BEP','BZ','BNO','CLNE','CRGY','CWEN','EFT','EQT','FSLR','GEV','NEE','NG','NGS','RNW','SM']

data = yf.download(tickers, start='2024-01-01', end='2026-11-28', auto_adjust=True, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    prices = data['Close'].copy()
else:
    prices = data.copy()
prices = prices.dropna(axis=1, how='all')
print("Missing tickers:", [t for t in tickers if t not in prices.columns])

returns = prices.pct_change(fill_method=None).dropna()
returns = returns.dropna(axis=1, how='any').dropna()
asset_names = returns.columns.tolist()
n = len(asset_names)
mu = returns.mean().values * 252
Sigma = returns.cov().values * 252
Sigma = (Sigma + Sigma.T) / 2 + np.eye(n) * 1e-8

def portfolio_return(w): return w @ mu
def portfolio_risk(w): return np.sqrt(max(w @ Sigma @ w, 0))

bounds = [(0, 1)] * n
initial = np.ones(n) / n
budget = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

minimum = minimize(lambda w: w @ Sigma @ w, initial, method='SLSQP', bounds=bounds, constraints=[budget])
w_min = minimum.x

risk_free_rate = 0.0
def negative_sharpe(w):
    risk = portfolio_risk(w)
    return 1e10 if risk < 1e-12 else -(portfolio_return(w) - risk_free_rate) / risk

maximum = minimize(negative_sharpe, w_min, method='SLSQP', bounds=bounds, constraints=[budget])
w_max = maximum.x

targets = np.linspace(portfolio_return(w_min), mu.max(), 100)
frontier_risk, frontier_return = [], []
for target in targets:
    res = minimize(lambda w: w @ Sigma @ w, w_min, method='SLSQP', bounds=bounds, constraints=[budget, {'type':'eq','fun': lambda w, t=target: portfolio_return(w)-t}])
    if res.success:
        frontier_risk.append(portfolio_risk(res.x)); frontier_return.append(portfolio_return(res.x))

def statistics(w):
    ret, risk = portfolio_return(w), portfolio_risk(w)
    return ret, risk, (ret - risk_free_rate)/risk

results = pd.DataFrame({'Asset': asset_names, 'Maximum Sharpe': w_max, 'Minimum Volatility': w_min})
print("\nPORTFOLIO WEIGHTS"); print(results.to_string(index=False))
print("\nPORTFOLIO PERFORMANCE")
for name, w in [("Maximum Sharpe", w_max), ("Minimum Volatility", w_min)]:
    ret, risk, sharpe = statistics(w)
    print(f"\n{name}\nExpected return: {ret:.4%}\nVolatility: {risk:.4%}\nSharpe ratio: {sharpe:.4f}")

plt.figure(figsize=(10,7))
plt.plot(frontier_risk, frontier_return, linewidth=2, label='Efficient Frontier')
r_min, v_min, _ = statistics(w_min); r_max, v_max, _ = statistics(w_max)
plt.scatter(v_min, r_min, marker='*', s=200, label='Minimum Volatility')
plt.scatter(v_max, r_max, marker='D', s=90, label='Maximum Sharpe')
plt.xlabel('Annualized Risk (Standard Deviation)'); plt.ylabel('Annualized Expected Return')
plt.title('Mean-Variance Efficient Frontier (2024-2026)'); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig('mean_variance_efficient_frontier.png', dpi=300, bbox_inches='tight')
plt.show()
results.to_csv('mean_variance_weights.csv', index=False)
print("\nAnalysis completed.")
