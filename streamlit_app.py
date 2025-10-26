import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm

RISK_FREE_RATE = 0.06  # Annual risk-free rate
MONTHS_IN_YEAR = 12

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Quant India',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_market_data():
    """
    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    start_date = "2025-09-01"
    end_date = datetime.today().strftime('%Y-%m-%d')


    ticker = '^CRSLDX'
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    last_date_data = data.index[-1]
    # Keep only the closing prices
    data = data[['Close']]
    market_data = data['Close'].resample('ME').last()

    return market_data, last_date_data


@st.cache_data
def get_portfolio_data(month_start):
    sheet_id = "10cMWuCXMb5-7tgaHWS5Ef-D0rNNhWSvgElVnY8f4t2c"
    sheet_name = "Holdings"  # or your specific sheet name
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    portfolio_data = pd.read_csv(url)
    portfolio_data['Portfolio Date'] = pd.to_datetime(portfolio_data['Portfolio Date'], format="%d-%m-%Y")
    portfolio_data = portfolio_data[portfolio_data['Factor Model Version'] == 2.0]

    # Group data by month
    return portfolio_data

# 1. Annualized Return
def annualized_return(monthly_returns):
    compounded = (1 + monthly_returns).prod()
    n_months = len(monthly_returns)
    return compounded ** (12 / n_months) - 1

# 2. Annualized Volatility
def annualized_volatility(monthly_returns):
    return monthly_returns.std(ddof=1) * np.sqrt(12)

# 3. Sharpe Ratio
def sharpe_ratio(ann_return, ann_volatility):
    excess_return = ann_return - RISK_FREE_RATE
    return excess_return / ann_volatility

def get_alpha_beta(portfolio_returns, benchmark_returns):
    # Prepare the returns (align index if needed)
    y = portfolio_returns.values
    x = benchmark_returns.values

    # Add constant for alpha
    x_with_const = sm.add_constant(x)

    # Run regression: portfolio return ~ alpha + beta * benchmark return
    model = sm.OLS(y, x_with_const).fit()
    alpha, beta = model.params
    return alpha, beta


# Convert to the first day of the current month
# Get necessary dates
today = datetime.today()
month_start = pd.to_datetime(today).replace(day=1).strftime("%d-%m-%Y")
today_month_name = pd.to_datetime(today).replace(day=1).strftime("%B")
today_year = pd.to_datetime(today).replace(day=1).strftime("%Y")

portfolio_data  = get_portfolio_data(month_start)
current_portfolio = portfolio_data[
    (portfolio_data['Portfolio Date'].dt.year == today.year) &
    (portfolio_data['Portfolio Date'].dt.month == today.month)
]

return_df = portfolio_data[portfolio_data['Quant Portfolio Return'].notna()]
# Calculate metrics for Portfolio
portfolio_returns = return_df['Quant Portfolio Return']
portfolio_ann_return = annualized_return(portfolio_returns)
portfolio_volatility = annualized_volatility(portfolio_returns)
portfolio_sharpe = sharpe_ratio(portfolio_ann_return, portfolio_volatility)

# Calculate metrics for NIFTY 500
benchmark_returns = return_df['NIFTY 500 Portfolio Return']
benchmark_ann_return = annualized_return(benchmark_returns)
benchmark_volatility = annualized_volatility(benchmark_returns)
benchmark_sharpe = sharpe_ratio(benchmark_ann_return, benchmark_volatility)

alpha, beta = get_alpha_beta(portfolio_returns, benchmark_returns)

# Create initial row with value 100
initial_date = return_df['Portfolio Date'].min() - pd.DateOffset(months=1)
initial_data = pd.DataFrame({
    'Portfolio Value': [100],
    'NIFTY 500 Portfolio Value': [100]
}, index=[initial_date])

# Calculate returns starting from 100
chart_df = pd.DataFrame({
    'Portfolio Value': 100 * (1 + return_df['Quant Portfolio Return']).cumprod(),
    'NIFTY 500 Portfolio Value': 100 * (1 + return_df['NIFTY 500 Portfolio Return']).cumprod()
})

# Set index and concatenate with initial row
chart_df.index = return_df['Portfolio Date']
chart_df = pd.concat([initial_data, chart_df])

### Get market (NIFTY 500) data
market_data, last_date_data = get_market_data()

# Display portfolio table
st.title("Quant India - Momentum + Quality Systematic Equities Strategy")
st.markdown(
    "Long-only exposure to Indian equities that captures beta and generates alpha using active factor tilts"
)

st.subheader("Methodology")
st.markdown("Version 2.0 - Updated Aug 2025")
st.markdown(
    "The quantitative factor portfolio methodology is a rule-based algorithm that creates an equal weighted portfolio of 20 stocks which is updated every month. These stocks are ranked and picked based on momentum and quality signals."
)
st.markdown('<a href="https://quantindia.substack.com/p/systematic-equities-strategy-shortcomings" target="_blank">View Methodology Details</a>', unsafe_allow_html=True)


st.subheader(f"📋 Current Portfolio ({today_month_name})")
st.table(current_portfolio[['Holding', 'Weight']].reset_index(drop=True))
st.markdown('<a href="https://docs.google.com/spreadsheets/d/10cMWuCXMb5-7tgaHWS5Ef-D0rNNhWSvgElVnY8f4t2c/edit?usp=drive_link" target="_blank">View Full Historical Portfolio</a>', unsafe_allow_html=True)


# Portfolio Performance Chart
st.subheader('Portfolio Performance', divider='gray')
st.line_chart(chart_df)

# Display metrics
st.subheader(f"Performance Metrics - From Aug 2025 to {today_month_name} {today_year}")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Portfolio")
    st.metric("Annualized Return", f"{portfolio_ann_return:.2%}")
    st.metric("Volatility", f"{portfolio_volatility:.2%}")
    st.metric("Sharpe Ratio", f"{portfolio_sharpe:.2f}")
    st.metric("Alpha", f"{alpha:.2%}")
    st.metric("Beta", f"{beta:.2f}")

with col2:
    st.markdown("### NIFTY 500 Benchmark")
    st.metric("Annualized Return", f"{benchmark_ann_return:.2%}")
    st.metric("Volatility", f"{benchmark_volatility:.2%}")
    st.metric("Sharpe Ratio", f"{benchmark_sharpe:.2f}")








