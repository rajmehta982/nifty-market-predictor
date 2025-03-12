import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from dateutil.relativedelta import relativedelta

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='NIFTY 50 Predictor',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_market_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/NIFTY 50_Historical_PR_01021990to06022025.csv'
    USD_INR_FILENAME = Path(__file__).parent/'data/USD_INR.csv'
    start_date = "1980-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    nifty_historical = pd.read_csv(DATA_FILENAME)
    nifty_historical['Date'] = pd.to_datetime(nifty_historical['Date'])
    nifty_historical = nifty_historical.set_index('Date').resample('ME').ffill()
    nifty_historical = nifty_historical[:'2007-10-01']
    nifty_historical  = pd.DataFrame(nifty_historical['Close'])
    nifty_historical = nifty_historical.rename(columns = {'Close': '^NSEI'})

    ticker = '^NSEI'
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    last_date_data = data.index[-1]
    # Keep only the closing prices
    data = data[['Close']]
    yf_market_data = data['Close'].resample('ME').last()
    market_data = pd.concat([nifty_historical, yf_market_data], axis=0)

    #CAPE Data
    sheet_id = "1ZcT4v4PzjwACcbzmwYXGwxLMRXtuQGwsSEPyONMcfcU"
    sheet_name = "Sheet1"  # or your specific sheet name
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    cape = pd.read_csv(url)
    cape['Date'] = pd.to_datetime(cape['Date'])
    cape = cape.set_index('Date').resample('ME').last()
    cape.head()
    market_data = pd.merge(market_data,cape['BSE Sensex CAPE 5'], left_index=True, right_index=True, how='left')

    # USD INR Exchange Data
    fred_data = pd.read_csv(USD_INR_FILENAME)
    fred_data['Date'] = pd.to_datetime(fred_data['Date'], format='%d-%m-%Y')
    fred_data = fred_data.set_index('Date').resample('ME').last()
    fred_data = fred_data[:'2003-11-30']
    fred_data  = pd.DataFrame(fred_data['USD/INR'])
    fred_data = fred_data.rename(columns = {'USD/INR': 'USDINR=X'})  
    # Download data
    usd_inr = yf.download('USDINR=X', start=start_date, end=end_date, interval="1d")

    # Keep only the closing prices
    usd_inr = usd_inr[['Close']]
    usd_inr = usd_inr.resample('ME').last()
    usd_inr = pd.concat([fred_data,usd_inr['Close']], axis=0)
    market_data = pd.merge(market_data, usd_inr['USDINR=X'], left_index=True, right_index=True, how='left')

    return market_data, last_date_data

def create_momentum_features(market_data):
    windows = [1, 3, 6, 9, 12]

    market_data['momentum_1_1'] = market_data['^NSEI'] / market_data['^NSEI'].shift(1)
    market_data['momentum_3_1'] = market_data['^NSEI'] / market_data['^NSEI'].shift(3)
    market_data['momentum_6_1'] = market_data['^NSEI'] / market_data['^NSEI'].shift(6)
    market_data['momentum_9_1'] = market_data['^NSEI'] / market_data['^NSEI'].shift(9)
    market_data['momentum_12_1'] = market_data['^NSEI'] / market_data['^NSEI'].shift(12)

    for window in windows:
        column_name = 'momentum_' + str(window) + '_USDINR'
        market_data[column_name] = market_data['USDINR=X'] / market_data['USDINR=X'].shift(window)

def train_model(market_data):
    initial_value = 100
    market_data['Return'] = market_data['^NSEI'].pct_change()
    market_data['Portfolio'] = initial_value * (1 + market_data['Return']).cumprod()
    market_data['Portfolio'].iloc[0] = initial_value
    market_data['next_month_return'] = market_data['Return'].shift(1)
    market_data.dropna(inplace=True)
    market_data['positive_returns'] = (market_data['next_month_return'] > -0.02).astype(int)

    training_columns = ['momentum_1_1',
       'momentum_3_1', 'momentum_6_1', 'momentum_9_1', 'momentum_12_1','BSE Sensex CAPE 5','momentum_1_USDINR', 'momentum_3_USDINR',
       'momentum_6_USDINR', 'momentum_9_USDINR', 'momentum_12_USDINR']

    X_final = market_data[training_columns].iloc[:-1]
    y_final = market_data['positive_returns'].iloc[:-1]

    # Initialize the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=10, random_state=42)

    # Define k-fold cross-validation (-fold)
    kfold = StratifiedKFold(n_splits=5, shuffle=False)

    # Perform cross-validation
    cv_scores = cross_val_score(clf, X_final, y_final, cv=kfold, scoring='accuracy')

    # Train the model
    clf.fit(X_final, y_final)

    return cv_scores, clf

    # Output results
    # print(f"Cross-Validation Scores: {cv_scores}")
    # print(f"Mean Accuracy: {np.mean(cv_scores):.4f}")
    # print(f"Standard Deviation: {np.std(cv_scores):.4f}")

def make_predictions(clf, market_data):
    training_columns = ['momentum_1_1',
       'momentum_3_1', 'momentum_6_1', 'momentum_9_1', 'momentum_12_1','BSE Sensex CAPE 5','momentum_1_USDINR', 'momentum_3_USDINR',
       'momentum_6_USDINR', 'momentum_9_USDINR', 'momentum_12_USDINR']
    last_month = np.array(market_data[training_columns].iloc[-1]).reshape(1,-1)
    final_prediction = clf.predict(last_month)

    return final_prediction

market_data, last_date_data = get_market_data()
create_momentum_features(market_data)
cv_scores, clf = train_model(market_data)
final_prediction = make_predictions(clf, market_data)

# Creating the UI
st.title("Momentum-based NIFTY 50 Prediction Model")
st.markdown(
    "A random forest model trained on momentum features created from 30+ years of monthly NIFTY 50 returns. It uses 1, 3, 6, 9, and 12 month momntum as predictiors for next month market returns. It predicts whether NIFTY 50 could fall by more than 2% next month."
)

# -----------------------------------------------------------------------------
# Draw the actual page
st.header('NIFTY 50 Historical Monthly Close Price (from July 1991)', divider='gray')
st.line_chart(
    market_data,
    y='^NSEI',
    color="#ffaa00",
)

last_date = market_data.index[-1]
current_month_name = last_date.strftime("%B")
# Get next month's datetime
next_month_dt = last_date + relativedelta(months=1)
# Get full month name
next_month_name = next_month_dt.strftime("%B")

st.header(f'Prediction for {next_month_name} end', divider='gray')

if final_prediction[0] == 0:
    st.metric(
        label='Prediction',
        value='Sell',
        
    )
else:
    st.metric(
        label='Prediction',
        value='Buy'
        
    )
st.markdown(
    "The buy or sell call tells whether the model predicts the market to fall more than 2% by the end of next month. The model is re-trained everyday with latest daily price for NIFTY and prediction is updated as well."
)


col1, col2 = st.columns(2)
with col1:
    st.header('Training Metrics', divider='gray')
    
    st.metric(
        label=f'Number of Months of Training Data',
        value=f'{len(market_data.iloc[:-1])}'
        
    )
    st.metric(
        label=f'Training Method',
        value=f'5-fold cross validation'
        
    )
    st.metric(
        label=f'Last Available Price On',
        value=last_date.strftime("%B %d, %Y")
        
    )

    st.metric(
        label=f'Mean Accuracy of Predictions',
        value=f'{np.mean(cv_scores)*100:.2f}%',
        
    )

    st.metric(
        label=f'Standard Deviation of Predictions',
        value=f'{np.std(cv_scores)*100:.2f}%',
        
    )

with col2:
    st.header('Model Info', divider='gray')
    st.metric(
        label=f'Model',
        value=f'Random Forest'
        
    )

    st.metric(
        label='Number of estimators',
        value=10
        
    )

    st.metric(
        label='Split Criterion',
        value='Gini'
        
    )

    st.metric(
        label='Features Used',
        value='Momentum, USD-INR Exchnage Rate, BSE 5 Years CAPE'
        
    )


    







