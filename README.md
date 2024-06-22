# Understanding Value at Risk (VaR): Historical, Monte Carlo, and Variance-Covariance Methods

## Introduction

Value at Risk (VaR) is a statistical measure used to assess the potential loss in value of an investment portfolio over a specified period for a given confidence interval. It answers the question: "What is my worst-case scenario loss over a certain time period with a certain level of confidence?" This analysis will explore three common methods to calculate VaR: Historical, Monte Carlo, and Variance-Covariance, using Python and data from five well-known companies: Apple (AAPL), Microsoft (MSFT), Google (GOOGL), Amazon (AMZN), and Facebook (META).

## Data Collection
We will fetch historical stock price data for the five companies using the yfinance library.

```python
import yfinance as yf
import pandas as pd

# List of stock symbols
stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# Fetch historical data
data = yf.download(stock_symbols, start='2020-01-01', end='2024-01-01')['Adj Close']
```

## Data Preprocessing
#### Why Calculate Returns?
Raw stock prices alone are not sufficient for VaR calculations because they do not directly reflect the changes in value over time. Instead, we calculate the daily returns, which represent the percentage change in price from one day to the next. This helps in standardizing the data and making it easier to model the risk. Returns provide a normalized measure of performance that accounts for the compounding effect and are essential for statistical analysis.

#### Calculating Returns
Daily returns are calculated as the percentage change in the adjusted closing price from one day to the next. This is done using the pct_change function, which computes the percentage change between the current and prior element.

#### Implementation
```python
# Calculate daily returns
returns = data.pct_change().dropna()

```

In this code:

- pct_change() calculates the daily returns.
- dropna() removes any missing values that result from the calculation, particularly the first row which will be NaN since there is no previous day to compare.

## Monte Carlo VaR
The Monte Carlo method involves simulating a large number of potential future price paths for the portfolio based on the statistical properties of the historical returns.

#### Formula
Simulate returns based on historical mean and standard deviation.
Calculate the VaR based on the simulated distribution.
#### Implementation

```python

import numpy as np

def monte_carlo_var(returns, num_simulations=1000, confidence_level=0.95):
    mean = returns.mean()
    std_dev = returns.std()
    simulated_returns = np.random.normal(mean, std_dev, (num_simulations, returns.shape[1]))
    var = np.percentile(simulated_returns, (1 - confidence_level) * 100, axis=0)
    return var

mc_var = monte_carlo_var(returns)
print(f"Monte Carlo VaR at 95% confidence level:\n{np.abs(mc_var)}")
#Result:
#Monte Carlo VaR at 95% confidence level:
#[-0.03231486 -0.03710083 -0.03554556 -0.0445836  -0.03171716]
```
## Historical VaR

The Historical VaR method is a non-parametric approach that involves sorting the historical returns and determining the VaR based on the desired confidence level.

#### Formula

VaR at confidence level $\alpha$:

$\text{VaR}_\alpha$ $=$ $\text{quantile} _{1-\alpha}(\text{returns})$

Where:

- $\text{VaR}_\alpha$ is the Value at Risk at confidence level $\alpha$.
- $\text{quantile}_{1-\alpha}(\text{returns})$ is the quantile of the returns 

#### Implementation
```python
def historical_var(returns, confidence_level=0.95):
    var = returns.quantile(1 - confidence_level)
    return var

hist_var = historical_var(returns)
print(f"Historical VaR at 95% confidence level:\n{hist_var.abs()}")
#Result:
#Historical VaR at 95% confidence level:
#Ticker
#AAPL    -0.032405
#AMZN    -0.035522
#GOOGL   -0.032451
#META    -0.041054
#MSFT    -0.029482
#Name: 0.050000000000000044, dtype: float64
```
## Variance-Covariance VaR

The Variance-Covariance method assumes that returns are normally distributed and uses the mean and covariance matrix of the returns to calculate VaR.

#### Formula

VaR for a portfolio at confidence level $\alpha$:
$\text{VaR} _\alpha$= $\mu$ + $\sigma$ $\cdot$ $z _\alpha$


Where:

- $\mu$ is the mean return.
- $\sigma$ is the standard deviation of returns.
- $Z_\alpha$ is the Z-score corresponding to the confidence level $\alpha$.

#### Implementation
```python
from scipy.stats import norm

def var_cov_var(returns, confidence_level=0.95):
    mean = returns.mean()
    cov_matrix = returns.cov()
    std_dev = np.sqrt(np.diag(cov_matrix))
    var = norm.ppf(1 - confidence_level, mean, std_dev)
    return var

vc_var = var_cov_var(returns)
print(f"Variance-Covariance VaR at 95% confidence level:\n{np.abs(vc_var)}")
#Results:
#Variance-Covariance VaR at 95% confidence level:
#[-0.03359537 -0.03830098 -0.03381249 -0.04750881 -0.03270021]
```


## Comparison and Conclusion
We compare the results from the three methods to understand the differences in the VaR calculations.

#### Visualization
![image](https://github.com/Deepti-Banger/Value-at-Risk-VaR-/assets/73093022/24fef867-d480-401d-89e1-a63fec2c35bc)

The comparison of Value at Risk (VaR) for AAPL, AMZN, GOOGL, META, and MSFT using Historical, Monte Carlo, and Variance-Covariance methods reveals the following insights:

### VaR Values
Monte Carlo VaR at 95% Confidence Level:
AAPL: -0.0323
AMZN: -0.0371
GOOGL: -0.0355
META: -0.0446
MSFT: -0.0317

### Historical VaR at 95% Confidence Level:
AAPL: -0.0324
AMZN: -0.0355
GOOGL: -0.0325
META: -0.0411
MSFT: -0.0295

### Variance-Covariance VaR at 95% Confidence Level:
AAPL: -0.0336
AMZN: -0.0383
GOOGL: -0.0338
META: -0.0475
MSFT: -0.0327

#### Key Observations
- Consistency Across Methods:

-The VaR values for each stock across all three methods are relatively close, indicating consistent risk assessment.
Magnitude of VaR:

- All VaR values are negative, as expected, indicating potential losses.
- Historical VaR generally provides slightly lower risk estimates compared to Monte Carlo and Variance-Covariance methods.

- Method Differences:

- Historical VaR: Based on actual historical data, tends to be more conservative for some stocks.
- Monte Carlo VaR: Uses simulations and provides a robust estimate by considering potential future scenarios.
- Variance-Covariance VaR: Assumes normally distributed returns, sometimes resulting in slightly higher risk estimates.

##### Stock-Specific Insights:

AAPL: All methods show very similar VaR values, with Historical VaR being slightly more conservative.
AMZN: Variance-Covariance method shows slightly higher risk compared to Historical and Monte Carlo.
GOOGL: Historical VaR is slightly lower, indicating slightly more conservative risk.
META: Shows the highest risk across all methods, with Variance-Covariance VaR being the highest.
MSFT: Historical VaR is the lowest, indicating a more conservative risk estimate.

## Summary
Overall, the analysis reveals that while all three methods provide consistent VaR estimates, the choice of method can slightly influence the VaR values. Historical VaR tends to be more conservative by capturing actual historical losses, whereas Monte Carlo and Variance-Covariance methods incorporate different assumptions and can sometimes show slightly higher or lower risk estimates.

Investors and risk managers should consider the context and assumptions of each method when choosing the appropriate VaR calculation for their portfolios. Understanding these differences helps in making more informed decisions regarding risk management and investment strategies.
