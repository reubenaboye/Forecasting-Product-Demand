########################
#title: Statistical Forecasts - Store Level
#######################
import numpy as np
import pandas as pd
from cProfile import label
from itertools import product
from typing import Union
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox

####################################### I. PRELIMINARY ###################################################################
# 1. Import Data==================================================

# 1.1. Store Data
CA_1_dta = pd.read_csv('Data/CA_Store1.csv', sep='\t')
CA_1_dta = CA_1_dta.drop('Unnamed: 0', axis=1)
CA_2_dta = pd.read_csv('Data/CA_Store2.csv', sep='\t')
CA_2_dta = CA_2_dta.drop('Unnamed: 0', axis=1)
CA_3_dta = pd.read_csv('Data/CA_Store3.csv',sep='\t')
CA_2_dta = CA_2_dta.drop('Unnamed: 0', axis=1)

TX_1_dta = pd.read_csv('Data/TX_Store1.csv',sep='\t')
TX_1_dta = TX_1_dta.drop('Unnamed: 0', axis=1)
TX_2_dta = pd.read_csv('Data/TX_Store2.csv',sep='\t')
TX_2_dta = TX_2_dta.drop('Unnamed: 0', axis=1)
TX_3_dta = pd.read_csv('Data/TX_Store3.csv', sep='\t')
TX_3_dta = TX_3_dta.drop('Unnamed: 0', axis=1)

WI_1_dta = pd.read_csv('Data/WI_Store1.csv', sep='\t')
WI_1_dta = WI_1_dta.drop('Unnamed: 0', axis=1)
WI_2_dta = pd.read_csv('Data/WI_Store2.csv', sep='\t')
WI_2_dta = CA_2_dta.drop('Unnamed: 0', axis=1)
WI_3_dta = pd.read_csv('Data/WI_Store3.csv', sep='\t')
WI_3_dta = WI_3_dta.drop('Unnamed: 0', axis=1)


# 2. Functions ========================================================

# 2.1. Hyperparameter Tuning Functions
def optimize_ARMA(endog: Union[pd.Series, list], order_list: list) -> pd.DataFrame:
    
    results = [] # initialize an empty list to sotre the order (p,q) and itrs correspnding AIC as a tuple

    for order in tqdm_notebook(order_list): # iterate over each (p,q) combination

        model = SARIMAX(endog, order=(order[0], 0, order[1]), simple_differencing=False).fit(disp=False) # Fit an ARIMA(p,q) model using SARIMAX function. We set set simple_differencing=False to presevent differencing. 

        aic = model.aic # calculae model's AIC
        results.append([order, aic])
    
    results_df = pd.DataFrame(results) # store (p,q) combination and AIC in df
    results_df.columns = ['(p,q)', 'AIC']

    # sort in ascending order, lower AIC is better
    results_df = results_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

    return results_df

def optimize_ARIMA(endog: Union[pd.Series, list], order_list: list, d: int) -> pd.DataFrame:
    results = [] # initialize an empty list to store each order (p,q) and corresponding AIC as tuple
    for order in tqdm_notebook(order_list):
        model = SARIMAX(endog, order=(order[0], d, order[1]), simple_differencing=False).fit(disp=False) # git am ARIMA(p,d,q) 
        aic = model.aic # calculate model's AIC
        results.append([order, aic]) # append (p,q) combination and AIC as a tuple to the results lsit
    results_df = pd.DataFrame(results)
    results_df.columns = ['(p,q)', 'AIC']

    #sort in ascending order, lower AIC bettween
    results_df = results_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return results_df

def optimize_ARIMA(endog: Union[pd.Series, list], order_list: list, d: int) -> pd.DataFrame:
    results = [] # initialize an empty list to store each order (p,q) and corresponding AIC as tuple
    for order in tqdm_notebook(order_list):
        model = SARIMAX(endog, order=(order[0], d, order[1]), simple_differencing=False).fit(disp=False) # git am ARIMA(p,d,q) 
        aic = model.aic # calculate model's AIC
        results.append([order, aic]) # append (p,q) combination and AIC as a tuple to the results lsit
    results_df = pd.DataFrame(results)
    results_df.columns = ['(p,q)', 'AIC']

    #sort in ascending order, lower AIC bettween
    results_df = results_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return results_df

def optimize_SARIMA(endog: Union[pd.Series, list], order_list: list, d: int, D: int, s: int) -> pd.DataFrame:
    results = []
    for order in tqdm_notebook(order_list):
        model = SARIMAX(endog,
        order=(order[0], d, order[1]),
        seasonal_order=(order[2], D, order[3], s),
        simple_differencing=False, initialization='approximate_diffuse').fit(disp=False)
        aic = model.aic
        results.append([order, aic])
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q,P,Q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return result_df


def optimize_SARIMA(endog: Union[pd.Series, list], order_list: list, d: int, D: int, s: int) -> pd.DataFrame:
    results = []
    for order in tqdm_notebook(order_list):
        model = SARIMAX(endog,
        order=(order[0], d, order[1]),
        seasonal_order=(order[2], D, order[3], s),
        simple_differencing=False, initialization='approximate_diffuse').fit(disp=False)
        aic = model.aic
        results.append([order, aic])
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q,P,Q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return result_df

def optimize_SARIMAX(endog: Union[pd.Series, list], exog: Union[pd.Series, list], order_list: list, d: int, D: int, s: int) -> pd.DataFrame:
    results = []

    for order in tqdm_notebook(order_list):
        model = SARIMAX(endog, exog, order=(order[0], d, order[1]),
        seasonal_order=(order[2], D, order[3], s),
        simple_differencing=False, initialization='approximate_diffuse').fit(disp=False)

        aic = model.aic
        results.append([order, aic])
    results_df = pd.DataFrame(results)
    results_df.columns = ['(p,q,P,Q)', 'AIC']

    # sort in ascending order, lower AiC better
    results_df = results_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

    return results_df

# 2.2 .Forecasting Functions

def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int, window: int, method:str) -> list:
    total_len = train_len + horizon
    if method == 'mean':
        pred_mean = []

        for i in range(train_len, total_len, window):
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))
        return pred_mean
    elif method == 'last':
        pred_last_value = []
        for i in range (train_len, total_len, window):
            last_value = df[:i].iloc[-1].values[0]
            pred_last_value.extend(last_value for _ in range(window))
        return pred_last_value
    elif method == 'MA':
        pred_MA = []
        for i in range(train_len, total_len, window):
            model = SARIMAX(df[:i], order=(0,0,2))
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window - 1)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_MA.extend(oos_pred)
        return pred_MA 
    elif method =='AR':
      pred_AR = []
      for i in range(train_len, total_len, window):
        model=SARIMAX(df[:i], order=(3,0,0))
        res=model.fit(disp=False)
        predictions=res.get_prediction(0, i + window - 1)
        oss_pred = predictions.predicted_mean.iloc[-window:]
        pred_AR.extend(oos_pred)
      return pred_AR
    elif method =='ARMA':
      pred_ARMA = []
      for i in range(train_len, total_len, window):
        model=SARIMAX(df[:i], order=(2,0,2))
        res=model.fit(disp=False)
        predictions = res.get_prediction(0, i + window - 1)
        oss_pred = predictions.predicted_mean.iloc[-window:]
        pred_ARMA.extend(oos_pred)
      return pred_ARMA

def rolling_forecast_SARIMA(endog: Union[pd.Series, list], exog: Union[pd.Series, list], train_len: int, horizon: int, window: int, method: str) -> list:
    total_len = train_len + horizon
    if method == 'last':
        pred_last_value = []
        
        for i in range(train_len, total_len, window):
            last_value = endog[:i].iloc[-1]
            pred_last_value.extend(last_value for _ in range(window))
        return pred_last_value
    elif method == 'SARIMAX':
        pred_SARIMAX = []
        
        for i in range(train_len, total_len, window):
            model = SARIMAX(endog[:i], exog[:i], order=(3,1,3), 
            seasonal_order=(0,0,0,4), simple_differencing=False)
            res = model.fit(disp=False)
            predictions = res.get_prediction(exog=exog)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_SARIMAX.extend(oos_pred)
    return pred_SARIMAX

######################### II. TIME SERIES ANALYSIS AND FORECASTS ##############################################################################

# 1. CALIFORNIA ==================================================================================================

# 1.1. CA_1 =========================

# Resample Data
CA_1_resample = CA_1_dta
CA_1_resample['yy-mm']  = CA_1_dta['Date'].dt.to_period('M')
CA_1_resample = CA_1_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
CA_1_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
CA_1_resample['Date'] = CA_1_resample['Date'].dt.to_timestamp()
CA_1_resample 

# 1.1.1. TIME SERIES ANALYSIS ------------------------------------------------------------------

# 1. Visualize Time Series Data
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(CA_1_resample['Date'], CA_1_resample['Sales']) # plot avg. weekly foot traffic
ax.set_xlabel('Date') # label x-axis
ax.set_ylabel('Sales') # label y-axis

#plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2)) # label ticks on x-axis
fig.autofmt_xdate() # tilt labels on x-axis ticks so they display nicely
plt.tight_layout() # remove whitespace around figure

# 2. Time Series Decomposition
CA_1_decomp = STL(CA_1_resample['Sales'], period=12).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(CA_1_decomp.observed)
ax1.set_ylabel('Observed')

ax2.plot(CA_1_decomp.trend)
ax2.set_ylabel('Trend')

ax3.plot(CA_1_decomp.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(CA_1_decomp.resid)
ax4.set_ylabel('Residuals')

#plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
fig.autofmt_xdate()
plt.tight_layout()

# 3. Test and Transform for Stationarity 
ad_fuller_res = adfuller(CA_1_resample['Sales'])
print(f'ADF Statistic: {ad_fuller_res[0]}')
print(f'p-value: {ad_fuller_res[1]}')

sales_diff = np.diff(CA_1_resample['Sales'], n=1)
ad_fuller_result = adfuller(sales_diff[1:])
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')

# 4. Correlation Analysis

#plot autocoreeleation 
plot_acf(sales_diff, lags=20);
#plot partial autocorrelation
plot_pacf(sales_diff, lags=20);

# 1.1.2. Model Fitting and Selection --------------------------------------------------------------------------------------------

# 1. Setup
#train-test split
train =  CA_1_resample['Sales'][:60]
test = CA_1_resample['Sales'].iloc[60:]
exog = CA_1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']]
exog_train = CA_1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[:60]
exog_test = CA_1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[60:]

# 2. Autoregressive Moving Average Model (ARMA)

    # 2.1. Fit differing ARMA Models (Hyperparameter Tuning)
ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
result_df = optimize_ARMA(sales_diff.astype(float), order_list)
result_df

    # 2.2. select model w/ optimal parameters
model = SARIMAX(train.astype(float), order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)

    # 2.3. residual analysis

# Qualitative 
model_fit.plot_diagnostics(figsize=(10,8));

# Quantitaitve
residuals = model_fit.resid 
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 3.  Autoregressive Integrated Moving Average (ARMA)
    # 3.1. Hyperparamter Tuning
ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 1
order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train.astype(float), order_list, d)
result_df

    # 3.2. Select model w/ optimal parameters
ARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)

    # 3.3. residual analysis

# qualitative
ARIMA_model_fit.plot_diagnostics(figsize=(10,8));

# quantiative
residuals = ARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 4. Seasonal Autoregressive Integrated Moving Average Model (SARIMA)

    # 4.1. Model Selection
ps = range(0, 4, 1)
qs = range(0, 4, 1)
Ps = range(0, 4, 1)
Qs = range(0, 4, 1)
SARIMA_order_list = list(product(ps, qs, Ps, Qs))
d = 1
D = 0
s = 12

SARIMA_result_df = optimize_SARIMA(train.astype(float), SARIMA_order_list, d, D, s)
SARIMA_result_df
    # 4.2. Fit model w/ hyperparameter
SARIMAX_model = SARIMAX(train.astype(float), exog_train.astype(float), order=(0, 1, 1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMAX_fit = SARIMAX_model.fit(disp=False)
print(SARIMAX_fit.summary())

    # 4.3. residual analysis

SARIMAX_fit.plot_diagnostics(figsize=(10,8));

residuals = SARIMAX_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
lb_result

# 1.1.3. Model Forecasts ---------------------------------------------------------------------------------------------------

# 1. Initialize hyperparameters
target = CA_1_resample['Sales']
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

# 2. Forecasts
pred_mean = rolling_forecast(target, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_ARMA = rolling_forecast(target.astype(float), TRAIN_LEN, HORIZON, WINDOW, 'ARMA')
pred_ARIMA = ARIMA_model_fit.get_prediction(60,64).predicted_mean
pred_SARIMA = SARIMA_model_fit.get_prediction(60, 64).predicted_mean
pred_SARIMAX = rolling_forecast_SARIMA(target.astype(float), exog.astype(float), TRAIN_LEN, HORIZON, WINDOW,'SARIMAX')

# store forecasts in dataframe
pred_df = pd.DataFrame({'actual': test})
pred_df['mean'] = pred_mean
pred_df['ARMA'] = pred_ARMA 
pred_df['ARIMA'] = pred_ARIMA
pred_df['SARIMA'] = pred_SARIMA
pred_df['SARIMAX'] = pred_SARIMAX

# calculate errors
from sklearn.metrics import mean_squared_error

error_df = {}

error_df['SARIMAX'] = mean_squared_error(pred_df['actual'], pred_df['SARIMAX'], squared=False)
error_df['ARIMA'] = mean_squared_error(pred_df['actual'], pred_df['ARIMA'], squared=False)
error_df['SARIMA'] = mean_squared_error(pred_df['actual'], pred_df['SARIMA'], squared=False)

# 1.2. CA_2 ============================================================================================

# Resample Data
CA_2_resample = CA_2_dta
CA_2_resample['yy-mm']  = CA_2_dta['Date'].dt.to_period('M')
CA_2_resample = CA_2_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
CA_2_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
CA_2_resample['Date'] = CA_2_resample['Date'].dt.to_timestamp()
CA_2_resample 

# 1.2.1. TIME SERIES ANALYSIS ------------------------------------------------------------------

# 1. Visualize Time Series Data
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(CA_2_resample['Date'], CA_2_resample['Sales']) # plot avg. weekly foot traffic
ax.set_xlabel('Date') # label x-axis
ax.set_ylabel('Sales') # label y-axis

#plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2)) # label ticks on x-axis
fig.autofmt_xdate() # tilt labels on x-axis ticks so they display nicely
plt.tight_layout() # remove whitespace around figure

# 2. Time Series Decomposition
CA_2_decomp = STL(CA_2_resample['Sales'], period=12).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(CA_2_decomp.observed)
ax1.set_ylabel('Observed')

ax2.plot(CA_2_decomp.trend)
ax2.set_ylabel('Trend')

ax3.plot(CA_2_decomp.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(CA_2_decomp.resid)
ax4.set_ylabel('Residuals')

#plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
fig.autofmt_xdate()
plt.tight_layout()

# 3. Test and Transform for Stationarity 
ad_fuller_res = adfuller(CA_2_resample['Sales'])
print(f'ADF Statistic: {ad_fuller_res[0]}')
print(f'p-value: {ad_fuller_res[1]}')

sales_diff = np.diff(CA_2_resample['Sales'], n=1)
ad_fuller_result = adfuller(sales_diff[1:])
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')

# 4. Correlation Analysis

#plot autocoreeleation 
plot_acf(sales_diff, lags=20);
#plot partial autocorrelation
plot_pacf(sales_diff, lags=20);

# 1.2.2. Model Fitting and Selection --------------------------------------------------------------------------------------------

# 1. Setup
#train-test split
train =  CA_2_resample['Sales'][:60]
test = CA_2_resample['Sales'].iloc[60:]
exog = CA_2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']]
exog_train = CA_2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[:60]
exog_test = CA_2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[60:]

# 2. Autoregressive Moving Average Model (ARMA)

    # 2.1. Fit differing ARMA Models (Hyperparameter Tuning)
ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
result_df = optimize_ARMA(sales_diff.astype(float), order_list)
result_df

    # 2.2. select model w/ optimal parameters
model = SARIMAX(train.astype(float), order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)

    # 2.3. residual analysis

# Qualitative 
model_fit.plot_diagnostics(figsize=(10,8));

# Quantitaitve
residuals = model_fit.resid 
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 3.  Autoregressive Integrated Moving Average (ARMA)
    # 3.1. Hyperparamter Tuning
ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 1
order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train.astype(float), order_list, d)
result_df

    # 3.2. Select model w/ optimal parameters
ARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)

    # 3.3. residual analysis

# qualitative
ARIMA_model_fit.plot_diagnostics(figsize=(10,8));

# quantiative
residuals = ARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 4. Seasonal Autoregressive Integrated Moving Average Model (SARIMA)

    # 4.1. Model Selection
ps = range(0, 4, 1)
qs = range(0, 4, 1)
Ps = range(0, 4, 1)
Qs = range(0, 4, 1)
SARIMA_order_list = list(product(ps, qs, Ps, Qs))
d = 1
D = 0
s = 12

SARIMA_result_df = optimize_SARIMA(train.astype(float), SARIMA_order_list, d, D, s)
SARIMA_result_df
    # 4.2. Fit model w/ hyperparameter
SARIMAX_model = SARIMAX(train.astype(float), exog_train.astype(float), order=(0, 1, 1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMAX_fit = SARIMAX_model.fit(disp=False)
print(SARIMAX_fit.summary())

    # 4.3. residual analysis

SARIMAX_fit.plot_diagnostics(figsize=(10,8));

residuals = SARIMAX_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
lb_result

# 1.2.3. Model Forecasts ---------------------------------------------------------------------------------------------------

# 1. Initialize hyperparameters
target = CA_2_resample['Sales']
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

# 2. Forecasts
pred_mean = rolling_forecast(target, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_ARMA = rolling_forecast(target.astype(float), TRAIN_LEN, HORIZON, WINDOW, 'ARMA')
pred_ARIMA = ARIMA_model_fit.get_prediction(60,64).predicted_mean
pred_SARIMA = SARIMA_model_fit.get_prediction(60, 64).predicted_mean
pred_SARIMAX = rolling_forecast_SARIMA(target.astype(float), exog.astype(float), TRAIN_LEN, HORIZON, WINDOW,'SARIMAX')

# store forecasts in dataframe
pred_df = pd.DataFrame({'actual': test})
pred_df['mean'] = pred_mean
pred_df['ARMA'] = pred_ARMA 
pred_df['ARIMA'] = pred_ARIMA
pred_df['SARIMA'] = pred_SARIMA
pred_df['SARIMAX'] = pred_SARIMAX

# calculate errors
from sklearn.metrics import mean_squared_error

error_df = {}

error_df['SARIMAX'] = mean_squared_error(pred_df['actual'], pred_df['SARIMAX'], squared=False)
error_df['ARIMA'] = mean_squared_error(pred_df['actual'], pred_df['ARIMA'], squared=False)
error_df['SARIMA'] = mean_squared_error(pred_df['actual'], pred_df['SARIMA'], squared=False)

# 1.3. CA_3 ==================================================================================================================================

# Resample Data
CA_3_resample = CA_3_dta
CA_3_resample['yy-mm']  = CA_3_dta['Date'].dt.to_period('M')
CA_3_resample = CA_3_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
CA_3_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
CA_3_resample['Date'] = CA_3_resample['Date'].dt.to_timestamp()
CA_3_resample 

# 1.3.1. TIME SERIES ANALYSIS ------------------------------------------------------------------

# 1. Visualize Time Series Data
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(CA_1_resample['Date'], CA_1_resample['Sales']) # plot avg. weekly foot traffic
ax.set_xlabel('Date') # label x-axis
ax.set_ylabel('Sales') # label y-axis

#plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2)) # label ticks on x-axis
fig.autofmt_xdate() # tilt labels on x-axis ticks so they display nicely
plt.tight_layout() # remove whitespace around figure

# 2. Time Series Decomposition
CA_3_decomp = STL(CA_3_resample['Sales'], period=12).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(CA_3_decomp.observed)
ax1.set_ylabel('Observed')

ax2.plot(CA_3_decomp.trend)
ax2.set_ylabel('Trend')

ax3.plot(CA_3_decomp.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(CA_3_decomp.resid)
ax4.set_ylabel('Residuals')

#plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
fig.autofmt_xdate()
plt.tight_layout()

# 3. Test and Transform for Stationarity 
ad_fuller_res = adfuller(CA_1_resample['Sales'])
print(f'ADF Statistic: {ad_fuller_res[0]}')
print(f'p-value: {ad_fuller_res[1]}')

sales_diff = np.diff(CA_1_resample['Sales'], n=1)
ad_fuller_result = adfuller(sales_diff[1:])
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')

# 4. Correlation Analysis

#plot autocoreeleation 
plot_acf(sales_diff, lags=20);
#plot partial autocorrelation
plot_pacf(sales_diff, lags=20);

# 1.3.2. Model Fitting and Selection --------------------------------------------------------------------------------------------

# 1. Setup
#train-test split
train =  CA_1_resample['Sales'][:60]
test = CA_1_resample['Sales'].iloc[60:]
exog = CA_1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']]
exog_train = CA_1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[:60]
exog_test = CA_1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[60:]

# 2. Autoregressive Moving Average Model (ARMA)

    # 2.1. Fit differing ARMA Models (Hyperparameter Tuning)
ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
result_df = optimize_ARMA(sales_diff.astype(float), order_list)
result_df

    # 2.2. select model w/ optimal parameters
model = SARIMAX(train.astype(float), order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)

    # 2.3. residual analysis

# Qualitative 
model_fit.plot_diagnostics(figsize=(10,8));

# Quantitaitve
residuals = model_fit.resid 
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 3.  Autoregressive Integrated Moving Average (ARMA)
    # 3.1. Hyperparamter Tuning
ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 1
order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train.astype(float), order_list, d)
result_df

    # 3.2. Select model w/ optimal parameters
ARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)

    # 3.3. residual analysis

# qualitative
ARIMA_model_fit.plot_diagnostics(figsize=(10,8));

# quantiative
residuals = ARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 4. Seasonal Autoregressive Integrated Moving Average Model (SARIMA)

    # 4.1. Model Selection
ps = range(0, 4, 1)
qs = range(0, 4, 1)
Ps = range(0, 4, 1)
Qs = range(0, 4, 1)
SARIMA_order_list = list(product(ps, qs, Ps, Qs))
d = 1
D = 0
s = 12

SARIMA_result_df = optimize_SARIMA(train.astype(float), SARIMA_order_list, d, D, s)
SARIMA_result_df
    # 4.2. Fit model w/ hyperparameter
SARIMAX_model = SARIMAX(train.astype(float), exog_train.astype(float), order=(0, 1, 1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMAX_fit = SARIMAX_model.fit(disp=False)
print(SARIMAX_fit.summary())

    # 4.3. residual analysis

SARIMAX_fit.plot_diagnostics(figsize=(10,8));

residuals = SARIMAX_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
lb_result

# 1.3.3. Model Forecasts ---------------------------------------------------------------------------------------------------

# 1. Initialize hyperparameters
target = CA_3_resample['Sales']
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

# 2. Forecasts
pred_mean = rolling_forecast(target, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_ARMA = rolling_forecast(target.astype(float), TRAIN_LEN, HORIZON, WINDOW, 'ARMA')
pred_ARIMA = ARIMA_model_fit.get_prediction(60,64).predicted_mean
pred_SARIMA = SARIMA_model_fit.get_prediction(60, 64).predicted_mean
pred_SARIMAX = rolling_forecast_SARIMA(target.astype(float), exog.astype(float), TRAIN_LEN, HORIZON, WINDOW,'SARIMAX')

# store forecasts in dataframe
pred_df = pd.DataFrame({'actual': test})
pred_df['mean'] = pred_mean
pred_df['ARMA'] = pred_ARMA 
pred_df['ARIMA'] = pred_ARIMA
pred_df['SARIMA'] = pred_SARIMA
pred_df['SARIMAX'] = pred_SARIMAX

# calculate errors
from sklearn.metrics import mean_squared_error
error_df = {}

error_df['SARIMAX'] = mean_squared_error(pred_df['actual'], pred_df['SARIMAX'], squared=False)
error_df['ARIMA'] = mean_squared_error(pred_df['actual'], pred_df['ARIMA'], squared=False)
error_df['SARIMA'] = mean_squared_error(pred_df['actual'], pred_df['SARIMA'], squared=False)

# 2. TEXAS ======================================================================================================================

# 2.1. TX_1  =============================================

# Resample Data
TX_1_resample = TX_1_dta
TX_1_resample['yy-mm']  = TX_1_resample['Date'].dt.to_period('M')
TX_1_resample = TX_1_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
TX_1_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
TX_1_resample['Date'] = TX_1_resample['Date'].dt.to_timestamp()
TX_1_resample

# 2.1.1. TIME SERIES ANALYSIS ------------------------------------------------------------------

# 1. Visualize Time Series Data
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(TX_1_resample['Date'], TX_1_resample['Sales']) # plot avg. weekly foot traffic
ax.set_xlabel('Date') # label x-axis
ax.set_ylabel('Sales') # label y-axis

#plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2)) # label ticks on x-axis
fig.autofmt_xdate() # tilt labels on x-axis ticks so they display nicely
plt.tight_layout() # remove whitespace around figure

# 2. Time Series Decomposition
TX_1_decomp = STL(TX_1_resample['Sales'], period=12).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(TX_1_decomp.observed)
ax1.set_ylabel('Observed')

ax2.plot(TX_1_decomp.trend)
ax2.set_ylabel('Trend')

ax3.plot(TX_1_decomp.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(TX_1_decomp.resid)
ax4.set_ylabel('Residuals')

#plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
fig.autofmt_xdate()
plt.tight_layout()

# 3. Test and Transform for Stationarity 
ad_fuller_res = adfuller(TX_1_resample['Sales'])
print(f'ADF Statistic: {ad_fuller_res[0]}')
print(f'p-value: {ad_fuller_res[1]}')


# 4. Correlation Analysis

#plot autocoreeleation 
plot_acf(TX_1_resample['Sales'], lags=20);
#plot partial autocorrelation
plot_pacf(TX_1_resample['Sales'], lags=20);

# 2.1.2. Model Fitting and Selection --------------------------------------------------------------------------------------------

# 1. Setup
#train-test split
train =  TX_1_resample['Sales'][:60]
test = TX_1_resample['Sales'].iloc[60:]
exog = TX_1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']]
exog_train = TX_1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[:60]
exog_test = TX_1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[60:]

# 2. Autoregressive Moving Average Model (ARMA)

    # 2.1. Fit differing ARMA Models (Hyperparameter Tuning)
ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
result_df = optimize_ARMA(sales_diff.astype(float), order_list)
result_df

    # 2.2. select model w/ optimal parameters
model = SARIMAX(train.astype(float), order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)

    # 2.3. residual analysis

# Qualitative 
model_fit.plot_diagnostics(figsize=(10,8));

# Quantitaitve
residuals = model_fit.resid 
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 3.  Autoregressive Integrated Moving Average (ARMA)
    # 3.1. Hyperparamter Tuning
ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 1
order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train.astype(float), order_list, d)
result_df

    # 3.2. Select model w/ optimal parameters
ARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)

    # 3.3. residual analysis

# qualitative
ARIMA_model_fit.plot_diagnostics(figsize=(10,8));

# quantiative
residuals = ARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 4. Seasonal Autoregressive Integrated Moving Average Model (SARIMA)

    # 4.1. Model Selection
ps = range(0, 4, 1)
qs = range(0, 4, 1)
Ps = range(0, 4, 1)
Qs = range(0, 4, 1)
SARIMA_order_list = list(product(ps, qs, Ps, Qs))
d = 1
D = 0
s = 12

SARIMA_result_df = optimize_SARIMA(train.astype(float), SARIMA_order_list, d, D, s)
SARIMA_result_df
    # 4.2. Fit model w/ hyperparameter
SARIMAX_model = SARIMAX(train.astype(float), exog_train.astype(float), order=(0, 1, 1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMAX_fit = SARIMAX_model.fit(disp=False)
print(SARIMAX_fit.summary())

    # 4.3. residual analysis

SARIMAX_fit.plot_diagnostics(figsize=(10,8));

residuals = SARIMAX_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
lb_result

# 2.1.3. Model Forecasts ---------------------------------------------------------------------------------------------------

# 2. Initialize hyperparameters
target = TX_1_resample['Sales']
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

# 3. Forecasts
pred_mean = rolling_forecast(target, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_ARMA = rolling_forecast(target.astype(float), TRAIN_LEN, HORIZON, WINDOW, 'ARMA')
pred_ARIMA = ARIMA_model_fit.get_prediction(60,64).predicted_mean
pred_SARIMA = SARIMA_model_fit.get_prediction(60, 64).predicted_mean
pred_SARIMAX = rolling_forecast_SARIMA(target.astype(float), exog.astype(float), TRAIN_LEN, HORIZON, WINDOW,'SARIMAX')

# store forecasts in dataframe
pred_df = pd.DataFrame({'actual': test})
pred_df['mean'] = pred_mean
pred_df['ARMA'] = pred_ARMA 
pred_df['ARIMA'] = pred_ARIMA
pred_df['SARIMA'] = pred_SARIMA
pred_df['SARIMAX'] = pred_SARIMAX

# calculate errors
from sklearn.metrics import mean_squared_error
error_df = {}

error_df['SARIMAX'] = mean_squared_error(pred_df['actual'], pred_df['SARIMAX'], squared=False)
error_df['ARIMA'] = mean_squared_error(pred_df['actual'], pred_df['ARIMA'], squared=False)
error_df['SARIMA'] = mean_squared_error(pred_df['actual'], pred_df['SARIMA'], squared=False)

# 2.2. TX_2 ====================================================================================================

# Resample Data
TX_2_resample = TX_2_dta
TX_2_resample['yy-mm']  = TX_1_resample['Date'].dt.to_period('M')
TX_2_resample = TX_2_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
TX_2_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
TX_2_resample['Date'] = TX_2_resample['Date'].dt.to_timestamp()
TX_2_resample

# 2.1.1. TIME SERIES ANALYSIS ------------------------------------------------------------------

# 1. Visualize Time Series Data
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(TX_2_resample['Date'], TX_2_resample['Sales']) # plot avg. weekly foot traffic
ax.set_xlabel('Date') # label x-axis
ax.set_ylabel('Sales') # label y-axis

#plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2)) # label ticks on x-axis
fig.autofmt_xdate() # tilt labels on x-axis ticks so they display nicely
plt.tight_layout() # remove whitespace around figure

# 2. Time Series Decomposition
TX_2_decomp = STL(TX_2_resample['Sales'], period=12).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(TX_2_decomp.observed)
ax1.set_ylabel('Observed')

ax2.plot(TX_2_decomp.trend)
ax2.set_ylabel('Trend')

ax3.plot(TX_2_decomp.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(TX_2_decomp.resid)
ax4.set_ylabel('Residuals')

#plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
fig.autofmt_xdate()
plt.tight_layout()

# 3. Test and Transform for Stationarity 
ad_fuller_res = adfuller(TX_2_resample['Sales'])
print(f'ADF Statistic: {ad_fuller_res[0]}')
print(f'p-value: {ad_fuller_res[1]}')


# 4. Correlation Analysis

#plot autocoreeleation 
plot_acf(TX_2_resample['Sales'], lags=20);
#plot partial autocorrelation
plot_pacf(TX_2_resample['Sales'], lags=20);

# 2.2.2. Model Fitting and Selection --------------------------------------------------------------------------------------------

# 1. Setup
#train-test split
train =  TX_2_resample['Sales'][:60]
test = TX_2_resample['Sales'].iloc[60:]
exog = TX_2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']]
exog_train = TX_2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[:60]
exog_test = TX_2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[60:]

# 2. Autoregressive Moving Average Model (ARMA)

    # 2.1. Fit differing ARMA Models (Hyperparameter Tuning)
ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
result_df = optimize_ARMA(sales_diff.astype(float), order_list)
result_df

    # 2.2. select model w/ optimal parameters
model = SARIMAX(train.astype(float), order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)

    # 2.3. residual analysis

# Qualitative 
model_fit.plot_diagnostics(figsize=(10,8));

# Quantitaitve
residuals = model_fit.resid 
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 3.  Autoregressive Integrated Moving Average (ARMA)
    # 3.1. Hyperparamter Tuning
ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 1
order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train.astype(float), order_list, d)
result_df

    # 3.2. Select model w/ optimal parameters
ARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)

    # 3.3. residual analysis

# qualitative
ARIMA_model_fit.plot_diagnostics(figsize=(10,8));

# quantiative
residuals = ARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 4. Seasonal Autoregressive Integrated Moving Average Model (SARIMA)

    # 4.1. Model Selection
ps = range(0, 4, 1)
qs = range(0, 4, 1)
Ps = range(0, 4, 1)
Qs = range(0, 4, 1)
SARIMA_order_list = list(product(ps, qs, Ps, Qs))
d = 1
D = 0
s = 12

SARIMA_result_df = optimize_SARIMA(train.astype(float), SARIMA_order_list, d, D, s)
SARIMA_result_df
    # 4.2. Fit model w/ hyperparameter
SARIMAX_model = SARIMAX(train.astype(float), exog_train.astype(float), order=(0, 1, 1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMAX_fit = SARIMAX_model.fit(disp=False)
print(SARIMAX_fit.summary())

    # 4.3. residual analysis

SARIMAX_fit.plot_diagnostics(figsize=(10,8));

residuals = SARIMAX_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
lb_result

# 2.2.3. Model Forecasts ---------------------------------------------------------------------------------------------------

# 1. Initialize hyperparameters
target = TX_2_resample['Sales']
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

# 2. Forecasts
pred_mean = rolling_forecast(target, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_ARMA = rolling_forecast(target.astype(float), TRAIN_LEN, HORIZON, WINDOW, 'ARMA')
pred_ARIMA = ARIMA_model_fit.get_prediction(60,64).predicted_mean
pred_SARIMA = SARIMA_model_fit.get_prediction(60, 64).predicted_mean
pred_SARIMAX = rolling_forecast_SARIMA(target.astype(float), exog.astype(float), TRAIN_LEN, HORIZON, WINDOW,'SARIMAX')

# store forecasts in dataframe
pred_df = pd.DataFrame({'actual': test})
pred_df['mean'] = pred_mean
pred_df['ARMA'] = pred_ARMA 
pred_df['ARIMA'] = pred_ARIMA
pred_df['SARIMA'] = pred_SARIMA
pred_df['SARIMAX'] = pred_SARIMAX

# calculate errors
from sklearn.metrics import mean_squared_error
error_df = {}

error_df['SARIMAX'] = mean_squared_error(pred_df['actual'], pred_df['SARIMAX'], squared=False)
error_df['ARIMA'] = mean_squared_error(pred_df['actual'], pred_df['ARIMA'], squared=False)
error_df['SARIMA'] = mean_squared_error(pred_df['actual'], pred_df['SARIMA'], squared=False)


# 2.3. TX_3 ====================================================================================================================

# Resample Data
TX_3_resample = TX_3_dta
TX_3_resample['yy-mm']  = TX_3_resample['Date'].dt.to_period('M')
TX_3_resample = TX_3_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
TX_3_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
TX_3_resample['Date'] = TX_3_resample['Date'].dt.to_timestamp()
TX_3_resample

# 2.1.1. TIME SERIES ANALYSIS ------------------------------------------------------------------

# 1. Visualize Time Series Data
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(TX_3_resample['Date'], TX_3_resample['Sales']) # plot avg. weekly foot traffic
ax.set_xlabel('Date') # label x-axis
ax.set_ylabel('Sales') # label y-axis

#plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2)) # label ticks on x-axis
fig.autofmt_xdate() # tilt labels on x-axis ticks so they display nicely
plt.tight_layout() # remove whitespace around figure

# 2. Time Series Decomposition
TX_3_decomp = STL(TX_3_resample['Sales'], period=12).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(TX_3_decomp.observed)
ax1.set_ylabel('Observed')

ax2.plot(TX_3_decomp.trend)
ax2.set_ylabel('Trend')

ax3.plot(TX_3_decomp.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(TX_3_decomp.resid)
ax4.set_ylabel('Residuals')

#plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
fig.autofmt_xdate()
plt.tight_layout()

# 3. Test and Transform for Stationarity 
ad_fuller_res = adfuller(TX_3_resample['Sales'])
print(f'ADF Statistic: {ad_fuller_res[0]}')
print(f'p-value: {ad_fuller_res[1]}')


# 4. Correlation Analysis

#plot autocoreeleation 
plot_acf(TX_3_resample['Sales'], lags=20);
#plot partial autocorrelation
plot_pacf(TX_3_resample['Sales'], lags=20);

# 2.2.2. Model Fitting and Selection --------------------------------------------------------------------------------------------

# 1. Setup
#train-test split
train =  TX_2_resample['Sales'][:60]
test = TX_2_resample['Sales'].iloc[60:]
exog = TX_2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']]
exog_train = TX_2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[:60]
exog_test = TX_2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[60:]

# 2. Autoregressive Moving Average Model (ARMA)

    # 2.1. Fit differing ARMA Models (Hyperparameter Tuning)
ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
result_df = optimize_ARMA(sales_diff.astype(float), order_list)
result_df

    # 2.2. select model w/ optimal parameters
model = SARIMAX(train.astype(float), order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)

    # 2.3. residual analysis

# Qualitative 
model_fit.plot_diagnostics(figsize=(10,8));

# Quantitaitve
residuals = model_fit.resid 
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 3.  Autoregressive Integrated Moving Average (ARMA)
    # 3.1. Hyperparamter Tuning
ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 1
order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train.astype(float), order_list, d)
result_df

    # 3.2. Select model w/ optimal parameters
ARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)

    # 3.3. residual analysis

# qualitative
ARIMA_model_fit.plot_diagnostics(figsize=(10,8));

# quantiative
residuals = ARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 4. Seasonal Autoregressive Integrated Moving Average Model (SARIMA)

    # 4.1. Model Selection
ps = range(0, 4, 1)
qs = range(0, 4, 1)
Ps = range(0, 4, 1)
Qs = range(0, 4, 1)
SARIMA_order_list = list(product(ps, qs, Ps, Qs))
d = 1
D = 0
s = 12

SARIMA_result_df = optimize_SARIMA(train.astype(float), SARIMA_order_list, d, D, s)
SARIMA_result_df
    # 4.2. Fit model w/ hyperparameter
SARIMAX_model = SARIMAX(train.astype(float), exog_train.astype(float), order=(0, 1, 1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMAX_fit = SARIMAX_model.fit(disp=False)
print(SARIMAX_fit.summary())

    # 4.3. residual analysis

SARIMAX_fit.plot_diagnostics(figsize=(10,8));

residuals = SARIMAX_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
lb_result

# 2.3.3. Model Forecasts ---------------------------------------------------------------------------------------------------

# 1. Initialize hyperparameters
target = TX_3_resample['Sales']
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

# 2. Forecasts
pred_mean = rolling_forecast(target, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_ARMA = rolling_forecast(target.astype(float), TRAIN_LEN, HORIZON, WINDOW, 'ARMA')
pred_ARIMA = ARIMA_model_fit.get_prediction(60,64).predicted_mean
pred_SARIMA = SARIMA_model_fit.get_prediction(60, 64).predicted_mean
pred_SARIMAX = rolling_forecast_SARIMA(target.astype(float), exog.astype(float), TRAIN_LEN, HORIZON, WINDOW,'SARIMAX')

# store forecasts in dataframe
pred_df = pd.DataFrame({'actual': test})
pred_df['mean'] = pred_mean
pred_df['ARMA'] = pred_ARMA 
pred_df['ARIMA'] = pred_ARIMA
pred_df['SARIMA'] = pred_SARIMA
pred_df['SARIMAX'] = pred_SARIMAX

# calculate errors
from sklearn.metrics import mean_squared_error
error_df = {}

error_df['SARIMAX'] = mean_squared_error(pred_df['actual'], pred_df['SARIMAX'], squared=False)
error_df['ARIMA'] = mean_squared_error(pred_df['actual'], pred_df['ARIMA'], squared=False)
error_df['SARIMA'] = mean_squared_error(pred_df['actual'], pred_df['SARIMA'], squared=False)


# 3. Wisconsin Stores ==================================================================================================================================================================

# 3.1. WI_1 ==============================================================================================

# resample data
WI_1_resample = WI_1_dta
WI_1_resample['yy-mm']  = WI_1_resample['Date'].dt.to_period('M')
WI_1_resample = WI_1_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
WI_1_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
WI_1_resample['Date'] = WI_1_resample['Date'].dt.to_timestamp()
WI_1_resample

# 3.1.1. Time Series Analysis ----------------------------------------------------------------
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(WI_1_resample['Date'], WI_1_resample['Sales']) # plot avg. weekly foot traffic
ax.set_xlabel('Date') # label x-axis
ax.set_ylabel('Sales') # label y-axis

#plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2)) # label ticks on x-axis
fig.autofmt_xdate() # tilt labels on x-axis ticks so they display nicely
plt.tight_layout() # remove whitespace around figure

# 1. Time-Series Decomposition
WI_1_decomp = STL(WI_1_resample['Sales'], period=12).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(WI_1_decomp.observed)
ax1.set_ylabel('Observed')

ax2.plot(WI_1_decomp.trend)
ax2.set_ylabel('Trend')

ax3.plot(WI_1_decomp.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(WI_1_decomp.resid)
ax4.set_ylabel('Residuals')

#plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
fig.autofmt_xdate()
plt.tight_layout()

# 2. Test and Transform for stationarity
ad_fuller_res = adfuller(WI_1_resample['Sales'])
print(f'ADF Statistic: {ad_fuller_res[0]}')
print(f'p-value: {ad_fuller_res[1]}')

# 3. Correlation Analysis
plot_acf(WI_1_decomp['Sales'], lags=20);

plot_pacf(WI_1_decomp['Sales'], lags=20);

# 3.1.2. Model and Fitting Selection ----------------------------------------------------------

# 1. train-test split
train =  WI_1_resample['Sales'][:60]
test = WI_1_resample['Sales'].iloc[60:]
exog = WI_1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']]
exog_train = WI_1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[:60]
exog_test = WI_1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[60:]

# 2. Autoregressive Moving Average Model (ARMA)
    # 2.1. Hyperparameter Tuning
ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
result_df = optimize_ARMA(sales_diff.astype(float), order_list)
result_df
    # 2.2.  fit optimal parameters
model = SARIMAX(train.astype(float), order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)
     # 2.3. Residual analysis
# Qualitative analysis
model_fit.plot_diagnostics(figsize=(10,8));
# Quantiative analysis
residuals = model_fit.resid 
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 3. Autoregressive Integrated Moving Average (ARIMA)
    # 3.1. Hyperparameter Tuning
ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 1
order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train.astype(float), order_list, d)
result_df
    # 3.2. Fit optimal parameters
ARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)
    # 3.3. Residual Analysis
# qualitative analysis
ARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = ARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 4. Seasonal Autoregressive Integrated Moving Average (SARIMA)
    # 4.1. Hyperparameter Tuning
#initialize hyperparameters
ps = range(0, 4, 1)
qs = range(0, 4, 1)
Ps = range(0, 4, 1)
Qs = range(0, 4, 1)
SARIMA_order_list = list(product(ps, qs, Ps, Qs))
d = 1
D = 0
s = 12
# conduct grid search
SARIMA_result_df = optimize_SARIMA(train.astype(float), SARIMA_order_list, d, D, s)
SARIMA_result_df
    # 4.2. Fit optimal parameters
SARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMA_model_fit = SARIMA_model.fit(disp=False)
    # 4.3. Residual analysis
# qualitative analysis
SARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = SARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 5. SARIMAX
    # 5.1. Hyperparameter Tuning
p = range(0, 4, 1)
d = 1
q = range(0, 4, 1)
P = range(0, 4, 1)
D = 0
Q = range(0, 4, 1)
s = 12

parameters = product(p, q, P, Q)
parameters_list = list(parameters)
result_df = optimize_SARIMAX(train.astype(float), exog_train.astype(float), parameters_list, d, D, s)
result_df
    # 5.2. Fit optimal parameters
SARIMAX_model = SARIMAX(train.astype(float), exog_train.astype(float), order=(0, 1, 1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMAX_fit = SARIMAX_model.fit(disp=False)
print(SARIMAX_fit.summary())
    # 5.3. Residual analysis
# qualitative analysis
SARIMAX_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = SARIMAX_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
lb_result

# 3.1.3. Forecasting -----------------------------------------------------------------------------------------
target = WI_1_resample['Sales']
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

pred_mean = rolling_forecast(target, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_ARMA = rolling_forecast(target.astype(float), TRAIN_LEN, HORIZON, WINDOW, 'ARMA')
pred_ARIMA = ARIMA_model_fit.get_prediction(60,64).predicted_mean
pred_SARIMA = SARIMA_model_fit.get_prediction(60, 64).predicted_mean
pred_SARIMAX = rolling_forecast_SARIMA(target.astype(float), exog.astype(float), TRAIN_LEN, HORIZON, WINDOW,'SARIMAX')


# 3.2. WI_2 ==============================================================================================

# resample data
WI_2_resample = WI_2_dta
WI_2_resample['yy-mm']  = WI_2_resample['Date'].dt.to_period('M')
WI_2_resample = WI_1_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
WI_2_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
WI_2_resample['Date'] = WI_2_resample['Date'].dt.to_timestamp()
WI_2_resample

# 3.2.1. Time Series Analysis ----------------------------------------------------------------
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(WI_1_resample['Date'], WI_1_resample['Sales']) # plot avg. weekly foot traffic
ax.set_xlabel('Date') # label x-axis
ax.set_ylabel('Sales') # label y-axis

#plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2)) # label ticks on x-axis
fig.autofmt_xdate() # tilt labels on x-axis ticks so they display nicely
plt.tight_layout() # remove whitespace around figure

# 1. Time-Series Decomposition
WI_2_decomp = STL(WI_2_resample['Sales'], period=12).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(WI_2_decomp.observed)
ax1.set_ylabel('Observed')

ax2.plot(WI_2_decomp.trend)
ax2.set_ylabel('Trend')

ax3.plot(WI_2_decomp.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(WI_2_decomp.resid)
ax4.set_ylabel('Residuals')

#plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
fig.autofmt_xdate()
plt.tight_layout()

# 2. Test and Transform for stationarity
ad_fuller_res = adfuller(WI_2_resample['Sales'])
print(f'ADF Statistic: {ad_fuller_res[0]}')
print(f'p-value: {ad_fuller_res[1]}')

# 3. Correlation Analysis
plot_acf(WI_2_decomp['Sales'], lags=20);

plot_pacf(WI_2_decomp['Sales'], lags=20);

# 3.2.2. Model and Fitting Selection ----------------------------------------------------------

# 1. train-test split
train =  WI_2_resample['Sales'][:60]
test = WI_2_resample['Sales'].iloc[60:]
exog = WI_2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']]
exog_train = WI_2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[:60]
exog_test = WI_2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[60:]

# 2. Autoregressive Moving Average Model (ARMA)
    # 2.1. Hyperparameter Tuning
ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
result_df = optimize_ARMA(sales_diff.astype(float), order_list)
result_df
    # 2.2.  fit optimal parameters
model = SARIMAX(train.astype(float), order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)
     # 2.3. Residual analysis
# Qualitative analysis
model_fit.plot_diagnostics(figsize=(10,8));
# Quantiative analysis
residuals = model_fit.resid 
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 3. Autoregressive Integrated Moving Average (ARIMA)
    # 3.1. Hyperparameter Tuning
ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 1
order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train.astype(float), order_list, d)
result_df
    # 3.2. Fit optimal parameters
ARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)
    # 3.3. Residual Analysis
# qualitative analysis
ARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = ARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 4. Seasonal Autoregressive Integrated Moving Average (SARIMA)
    # 4.1. Hyperparameter Tuning
#initialize hyperparameters
ps = range(0, 4, 1)
qs = range(0, 4, 1)
Ps = range(0, 4, 1)
Qs = range(0, 4, 1)
SARIMA_order_list = list(product(ps, qs, Ps, Qs))
d = 1
D = 0
s = 12
# conduct grid search
SARIMA_result_df = optimize_SARIMA(train.astype(float), SARIMA_order_list, d, D, s)
SARIMA_result_df
    # 4.2. Fit optimal parameters
SARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMA_model_fit = SARIMA_model.fit(disp=False)
    # 4.3. Residual analysis
# qualitative analysis
SARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = SARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 5. SARIMAX
    # 5.1. Hyperparameter Tuning
p = range(0, 4, 1)
d = 1
q = range(0, 4, 1)
P = range(0, 4, 1)
D = 0
Q = range(0, 4, 1)
s = 12

parameters = product(p, q, P, Q)
parameters_list = list(parameters)
result_df = optimize_SARIMAX(train.astype(float), exog_train.astype(float), parameters_list, d, D, s)
result_df
    # 5.2. Fit optimal parameters
SARIMAX_model = SARIMAX(train.astype(float), exog_train.astype(float), order=(0, 1, 1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMAX_fit = SARIMAX_model.fit(disp=False)
print(SARIMAX_fit.summary())
    # 5.3. Residual analysis
# qualitative analysis
SARIMAX_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = SARIMAX_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
lb_result

# 3.2.3. Forecasting -----------------------------------------------------------------------------------------
target = WI_2_resample['Sales']
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

pred_mean = rolling_forecast(target, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_ARMA = rolling_forecast(target.astype(float), TRAIN_LEN, HORIZON, WINDOW, 'ARMA')
pred_ARIMA = ARIMA_model_fit.get_prediction(60,64).predicted_mean
pred_SARIMA = SARIMA_model_fit.get_prediction(60, 64).predicted_mean
pred_SARIMAX = rolling_forecast_SARIMA(target.astype(float), exog.astype(float), TRAIN_LEN, HORIZON, WINDOW,'SARIMAX')

# 3.3. WI_3 ==============================================================================================

# resample data
WI_3_resample = WI_3_dta
WI_3_resample['yy-mm']  = WI_3_resample['Date'].dt.to_period('M')
WI_3_resample = WI_3_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
WI_3_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
WI_3_resample['Date'] = WI_3_resample['Date'].dt.to_timestamp()
WI_3_resample

# 3.1.1. Time Series Analysis ----------------------------------------------------------------
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(WI_3_resample['Date'], WI_3_resample['Sales']) # plot avg. weekly foot traffic
ax.set_xlabel('Date') # label x-axis
ax.set_ylabel('Sales') # label y-axis

#plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2)) # label ticks on x-axis
fig.autofmt_xdate() # tilt labels on x-axis ticks so they display nicely
plt.tight_layout() # remove whitespace around figure

# 1. Time-Series Decomposition
WI_3_decomp = STL(WI_3_resample['Sales'], period=12).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(WI_3_decomp.observed)
ax1.set_ylabel('Observed')

ax2.plot(WI_3_decomp.trend)
ax2.set_ylabel('Trend')

ax3.plot(WI_3_decomp.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(WI_3_decomp.resid)
ax4.set_ylabel('Residuals')

#plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
fig.autofmt_xdate()
plt.tight_layout()

# 2. Test and Transform for stationarity
ad_fuller_res = adfuller(WI_3_resample['Sales'])
print(f'ADF Statistic: {ad_fuller_res[0]}')
print(f'p-value: {ad_fuller_res[1]}')

# 3. Correlation Analysis
plot_acf(WI_3_decomp['Sales'], lags=20);

plot_pacf(WI_3_decomp['Sales'], lags=20);

# 3.1.2. Model and Fitting Selection ----------------------------------------------------------

# 1. train-test split
train =  WI_3_resample['Sales'][:60]
test = WI_3_resample['Sales'].iloc[60:]
exog = WI_3_resample[['Unemployment', 'Inflation', 'Real Disposable Income']]
exog_train = WI_3_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[:60]
exog_test = WI_3_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[60:]

# 2. Autoregressive Moving Average Model (ARMA)
    # 2.1. Hyperparameter Tuning
ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
result_df = optimize_ARMA(sales_diff.astype(float), order_list)
result_df
    # 2.2.  fit optimal parameters
model = SARIMAX(train.astype(float), order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)
     # 2.3. Residual analysis
# Qualitative analysis
model_fit.plot_diagnostics(figsize=(10,8));
# Quantiative analysis
residuals = model_fit.resid 
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 3. Autoregressive Integrated Moving Average (ARIMA)
    # 3.1. Hyperparameter Tuning
ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 1
order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train.astype(float), order_list, d)
result_df
    # 3.2. Fit optimal parameters
ARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)
    # 3.3. Residual Analysis
# qualitative analysis
ARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = ARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 4. Seasonal Autoregressive Integrated Moving Average (SARIMA)
    # 4.1. Hyperparameter Tuning
#initialize hyperparameters
ps = range(0, 4, 1)
qs = range(0, 4, 1)
Ps = range(0, 4, 1)
Qs = range(0, 4, 1)
SARIMA_order_list = list(product(ps, qs, Ps, Qs))
d = 1
D = 0
s = 12
# conduct grid search
SARIMA_result_df = optimize_SARIMA(train.astype(float), SARIMA_order_list, d, D, s)
SARIMA_result_df
    # 4.2. Fit optimal parameters
SARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMA_model_fit = SARIMA_model.fit(disp=False)
    # 4.3. Residual analysis
# qualitative analysis
SARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = SARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 5. SARIMAX
    # 5.1. Hyperparameter Tuning
p = range(0, 4, 1)
d = 1
q = range(0, 4, 1)
P = range(0, 4, 1)
D = 0
Q = range(0, 4, 1)
s = 12

parameters = product(p, q, P, Q)
parameters_list = list(parameters)
result_df = optimize_SARIMAX(train.astype(float), exog_train.astype(float), parameters_list, d, D, s)
result_df
    # 5.2. Fit optimal parameters
SARIMAX_model = SARIMAX(train.astype(float), exog_train.astype(float), order=(0, 1, 1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMAX_fit = SARIMAX_model.fit(disp=False)
print(SARIMAX_fit.summary())
    # 5.3. Residual analysis
# qualitative analysis
SARIMAX_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = SARIMAX_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
lb_result

# 3.1.3. Forecasting -----------------------------------------------------------------------------------------
target = WI_3_resample['Sales']
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

pred_mean = rolling_forecast(target, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_ARMA = rolling_forecast(target.astype(float), TRAIN_LEN, HORIZON, WINDOW, 'ARMA')
pred_ARIMA = ARIMA_model_fit.get_prediction(60,64).predicted_mean
pred_SARIMA = SARIMA_model_fit.get_prediction(60, 64).predicted_mean
pred_SARIMAX = rolling_forecast_SARIMA(target.astype(float), exog.astype(float), TRAIN_LEN, HORIZON, WINDOW,'SARIMAX')



