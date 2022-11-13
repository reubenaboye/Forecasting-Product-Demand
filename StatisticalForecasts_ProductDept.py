########################
#title: Statistical Forecasts - Product Dept Level
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

foods1 = pd.read_csv('Product_Dept_Foods1.csv', sep='\t')
foods1 = foods1.drop('Unnamed: 0', axis=1)
foods2 = pd.read_csv('Product_Dept_Foods2.csv', sep='\t')
foods2 = foods2.drop('Unnamed: 0', axis=1)
foods3 = pd.read_csv('Product_Dept_Foods3.csv', sep='\t')
foods3 = foods3.drop('Unnamed: 0', axis=1)

household1 = pd.read_csv('Product_Dept_Household1.csv', sep='\t')
household1 = household1.drop('Unnamed: 0', axis=1)
household2 = pd.read_csv('Product_Dept_Household1.csv', sep='\t')
household2 = household2.drop('Unnamed: 0', axis=1)

hobbies1 = pd.read_csv('Product_Dept_Hobbies1.csv', sep='\t')
hobbies1 = hobbies1.drop('Unnamed: 0', axis=1)
hobbies2 = pd.read_csv('Product_Dept_Hobbies1.csv', sep='\t')
hobbies2 = hobbies2.drop('Unnamed: 0', axis=1)

# 2. Functions ==========================================================

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

# II.1. FOODS =======================================================================================

# 1. Foods1 ==============================================================================

# Resample data
foods1_resample = foods1
foods1_resample['yy-mm']  = foods1_resample['Date'].dt.to_period('M')
foods1_resample = foods1_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
foods1_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
foods1_resample['Date'] = foods1_resample['Date'].dt.to_timestamp()
foods1_resample 

# 1.1. Time Series Analysis-------------------------------------------------------
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(foods1_resample['Date'], foods1_resample['Sales']) # plot avg. weekly foot traffic
ax.set_xlabel('Date') # label x-axis
ax.set_ylabel('Sales') # label y-axis

#plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2)) # label ticks on x-axis
fig.autofmt_xdate() # tilt labels on x-axis ticks so they display nicely
plt.tight_layout() # remove whitespace around figure

# 1. Time Series Decomposition
foods1_decomp = STL(foods1_resample['Sales'], period=12).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(foods1_decomp.observed)
ax1.set_ylabel('Observed')

ax2.plot(foods1_decomp.trend)
ax2.set_ylabel('Trend')

ax3.plot(foods1_decomp.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(foods1_decomp.resid)
ax4.set_ylabel('Residuals')

#plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
fig.autofmt_xdate()
plt.tight_layout()

# 2. Test and Transform  Stationarity
ad_fuller_res = adfuller(foods1_resample['Sales'])
print(f'ADF Statistic: {ad_fuller_res[0]}')
print(f'p-value: {ad_fuller_res[1]}')

# 3. Correlation Analysis
plot_acf(foods1_resample['Sales'], lags=20);
plot_pacf(foods1_resample['Sales'], lags=20);

# 1.2. Model Fitting and Selection --------------------------------------------------------------------------

# 1. Setup
#train-test split
train =  foods1_resample['Sales'][:60]
test = foods1_resample['Sales'].iloc[60:]
exog = foods1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']]
exog_train = foods1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[:60]
exog_test = foods1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[60:]

# 2. Autoregressive Moving Average (ARMA)
    # 2.1. Grid search
ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
result_df = optimize_ARMA(train.astype(float), order_list)
result_df
    # 2.2. Fit optimal parameters
# Choose best model and fit on training data
model = SARIMAX(train.astype(float), order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)
    # 2.3. Residual analysis
# qualitiative analysis
model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = model_fit.resid 
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 3. ARIMA
    # 3.1. Grid Search 
ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 1
order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train.astype(float), order_list, d)
result_df
    # 3.2. Fit optimal parameters
ARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)
    # 3.3. Residual analysis
# qualitative analysis
ARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = ARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 4. SARIMA
    # 4.1. Grid Search
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
    # 4.2. Fit optimal parameters
SARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMA_model_fit = SARIMA_model.fit(disp=False)
    # 4.2. Residual analysis
# qualitative
SARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative
residuals = SARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 5. SARIMAX
    # 5.1. Grid Search
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
SARIMAX_fit.plot_diagnostics(figsize=(10,8));
residuals = SARIMAX_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
lb_result


# 1.3. Forecasts ------------------------------------------------------------------------------------------

# Initialize hyperparameter
target = foods1_resample['Sales']
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

pred_mean = rolling_forecast(target, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_ARMA = rolling_forecast(target.astype(float), TRAIN_LEN, HORIZON, WINDOW, 'ARMA')
pred_ARIMA = ARIMA_model_fit.get_prediction(60,64).predicted_mean
pred_SARIMA = SARIMA_model_fit.get_prediction(60, 64).predicted_mean
pred_SARIMAX = rolling_forecast_SARIMA(target.astype(float), exog.astype(float), TRAIN_LEN, HORIZON, WINDOW,'SARIMAX')



pred_df = pd.DataFrame({'actual': test})
pred_df['mean'] = pred_mean
pred_df['ARMA'] = pred_ARMA 
pred_df['ARIMA'] = pred_ARIMA
pred_df['SARIMA'] = pred_SARIMA
pred_df['SARIMAX'] = pred_SARIMAX


from sklearn.metrics import mean_squared_error

error_df = {}

error_df['SARIMAX'] = mean_squared_error(pred_df['actual'], pred_df['SARIMAX'], squared=False)
error_df['ARIMA'] = mean_squared_error(pred_df['actual'], pred_df['ARIMA'], squared=False)
error_df['SARIMA'] = mean_squared_error(pred_df['actual'], pred_df['SARIMA'], squared=False)
# 2. Foods2 ==============================================================================

# Resample data
foods2_resample = foods2
foods2_resample['yy-mm']  = foods2_resample['Date'].dt.to_period('M')
foods2_resample = foods2_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
foods2_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
foods2_resample['Date'] = foods2_resample['Date'].dt.to_timestamp()
foods2_resample 

# 2.1. Time Series Analysis-------------------------------------------------------
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(foods2_resample['Date'], foods2_resample['Sales']) # plot avg. weekly foot traffic
ax.set_xlabel('Date') # label x-axis
ax.set_ylabel('Sales') # label y-axis

#plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2)) # label ticks on x-axis
fig.autofmt_xdate() # tilt labels on x-axis ticks so they display nicely
plt.tight_layout() # remove whitespace around figure

# 1. Time Series Decomposition
foods1_decomp = STL(foods2_resample['Sales'], period=12).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(foods2_decomp.observed)
ax1.set_ylabel('Observed')

ax2.plot(foods2_decomp.trend)
ax2.set_ylabel('Trend')

ax3.plot(foods2_decomp.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(foods2_decomp.resid)
ax4.set_ylabel('Residuals')

#plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
fig.autofmt_xdate()
plt.tight_layout()

# 2. Test and Transform  Stationarity
ad_fuller_res = adfuller(foods2_resample['Sales'])
print(f'ADF Statistic: {ad_fuller_res[0]}')
print(f'p-value: {ad_fuller_res[1]}')

# 3. Correlation Analysis
plot_acf(foods2_resample['Sales'], lags=20);
plot_pacf(foods2_resample['Sales'], lags=20);

# 2.2. Model Fitting and Selection --------------------------------------------------------------------------

# 1. Setup
#train-test split
train =  foods2_resample['Sales'][:60]
test = foods2_resample['Sales'].iloc[60:]
exog = foods2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']]
exog_train = foods2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[:60]
exog_test = foods2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[60:]

# 2. Autoregressive Moving Average (ARMA)
    # 2.1. Grid search
ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
result_df = optimize_ARMA(train.astype(float), order_list)
result_df
    # 2.2. Fit optimal parameters
# Choose best model and fit on training data
model = SARIMAX(train.astype(float), order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)
    # 2.3. Residual analysis
# qualitiative analysis
model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = model_fit.resid 
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 3. ARIMA
    # 3.1. Grid Search 
ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 1
order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train.astype(float), order_list, d)
result_df
    # 3.2. Fit optimal parameters
ARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)
    # 3.3. Residual analysis
# qualitative analysis
ARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = ARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 4. SARIMA
    # 4.1. Grid Search
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
    # 4.2. Fit optimal parameters
SARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMA_model_fit = SARIMA_model.fit(disp=False)
    # 4.2. Residual analysis
# qualitative
SARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative
residuals = SARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 5. SARIMAX
    # 5.1. Grid Search
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
SARIMAX_fit.plot_diagnostics(figsize=(10,8));
residuals = SARIMAX_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
lb_result


# 2.3. Forecasts ------------------------------------------------------------------------------------------

# Initialize hyperparameter
target = foods2_resample['Sales']
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

pred_mean = rolling_forecast(target, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_ARMA = rolling_forecast(target.astype(float), TRAIN_LEN, HORIZON, WINDOW, 'ARMA')
pred_ARIMA = ARIMA_model_fit.get_prediction(60,64).predicted_mean
pred_SARIMA = SARIMA_model_fit.get_prediction(60, 64).predicted_mean
pred_SARIMAX = rolling_forecast_SARIMA(target.astype(float), exog.astype(float), TRAIN_LEN, HORIZON, WINDOW,'SARIMAX')



pred_df = pd.DataFrame({'actual': test})
pred_df['mean'] = pred_mean
pred_df['ARMA'] = pred_ARMA 
pred_df['ARIMA'] = pred_ARIMA
pred_df['SARIMA'] = pred_SARIMA
pred_df['SARIMAX'] = pred_SARIMAX


from sklearn.metrics import mean_squared_error

error_df = {}

error_df['SARIMAX'] = mean_squared_error(pred_df['actual'], pred_df['SARIMAX'], squared=False)
error_df['ARIMA'] = mean_squared_error(pred_df['actual'], pred_df['ARIMA'], squared=False)
error_df['SARIMA'] = mean_squared_error(pred_df['actual'], pred_df['SARIMA'], squared=False)

# 3. Foods3 ==============================================================================

# Resample data
foods3_resample = foods3
foods3_resample['yy-mm']  = foods3_resample['Date'].dt.to_period('M')
foods3_resample = foods3_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
foods3_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
foods3_resample['Date'] = foods3_resample['Date'].dt.to_timestamp()
foods3_resample 

# 1.1. Time Series Analysis-------------------------------------------------------
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(foods3_resample['Date'], foods3_resample['Sales']) # plot avg. weekly foot traffic
ax.set_xlabel('Date') # label x-axis
ax.set_ylabel('Sales') # label y-axis

#plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2)) # label ticks on x-axis
fig.autofmt_xdate() # tilt labels on x-axis ticks so they display nicely
plt.tight_layout() # remove whitespace around figure

# 1. Time Series Decomposition
foods3_decomp = STL(foods3_resample['Sales'], period=12).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(foods3_decomp.observed)
ax1.set_ylabel('Observed')

ax2.plot(foods3_decomp.trend)
ax2.set_ylabel('Trend')

ax3.plot(foods3_decomp.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(foods3_decomp.resid)
ax4.set_ylabel('Residuals')

#plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
fig.autofmt_xdate()
plt.tight_layout()

# 2. Test and Transform  Stationarity
ad_fuller_res = adfuller(foods3_resample['Sales'])
print(f'ADF Statistic: {ad_fuller_res[0]}')
print(f'p-value: {ad_fuller_res[1]}')

# 3. Correlation Analysis
plot_acf(foods1_resample['Sales'], lags=20);
plot_pacf(foods1_resample['Sales'], lags=20);

# 1.2. Model Fitting and Selection --------------------------------------------------------------------------

# 1. Setup
#train-test split
train =  foods3_resample['Sales'][:60]
test = foods3_resample['Sales'].iloc[60:]
exog = foods3_resample[['Unemployment', 'Inflation', 'Real Disposable Income']]
exog_train = foods3_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[:60]
exog_test = foods3_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[60:]

# 2. Autoregressive Moving Average (ARMA)
    # 2.1. Grid search
ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
result_df = optimize_ARMA(train.astype(float), order_list)
result_df
    # 2.2. Fit optimal parameters
# Choose best model and fit on training data
model = SARIMAX(train.astype(float), order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)
    # 2.3. Residual analysis
# qualitiative analysis
model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = model_fit.resid 
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 3. ARIMA
    # 3.1. Grid Search 
ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 1
order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train.astype(float), order_list, d)
result_df
    # 3.2. Fit optimal parameters
ARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)
    # 3.3. Residual analysis
# qualitative analysis
ARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = ARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 4. SARIMA
    # 4.1. Grid Search
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
    # 4.2. Fit optimal parameters
SARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMA_model_fit = SARIMA_model.fit(disp=False)
    # 4.2. Residual analysis
# qualitative
SARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative
residuals = SARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 5. SARIMAX
    # 5.1. Grid Search
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
SARIMAX_fit.plot_diagnostics(figsize=(10,8));
residuals = SARIMAX_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
lb_result


# 3.3. Forecasts ------------------------------------------------------------------------------------------

# Initialize hyperparameter
target = foods3_resample['Sales']
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

pred_mean = rolling_forecast(target, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_ARMA = rolling_forecast(target.astype(float), TRAIN_LEN, HORIZON, WINDOW, 'ARMA')
pred_ARIMA = ARIMA_model_fit.get_prediction(60,64).predicted_mean
pred_SARIMA = SARIMA_model_fit.get_prediction(60, 64).predicted_mean
pred_SARIMAX = rolling_forecast_SARIMA(target.astype(float), exog.astype(float), TRAIN_LEN, HORIZON, WINDOW,'SARIMAX')



pred_df = pd.DataFrame({'actual': test})
pred_df['mean'] = pred_mean
pred_df['ARMA'] = pred_ARMA 
pred_df['ARIMA'] = pred_ARIMA
pred_df['SARIMA'] = pred_SARIMA
pred_df['SARIMAX'] = pred_SARIMAX


from sklearn.metrics import mean_squared_error

error_df = {}

error_df['SARIMAX'] = mean_squared_error(pred_df['actual'], pred_df['SARIMAX'], squared=False)
error_df['ARIMA'] = mean_squared_error(pred_df['actual'], pred_df['ARIMA'], squared=False)
error_df['SARIMA'] = mean_squared_error(pred_df['actual'], pred_df['SARIMA'], squared=False)


# II.2. HOUSEHOLD =================================================================================

# 1. HOUSEHOLD1 ==========================================================
# Resample data
household1_resample = household1
household1_resample['yy-mm']  = household1_resample['Date'].dt.to_period('M')
household1_resample = household1_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
household1_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
household1_resample['Date'] = household1_resample['Date'].dt.to_timestamp()
household1_resample 

# 1.1. Time Series Analysis-------------------------------------------------------
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(household1_resample['Date'], household1_resample['Sales']) # plot avg. weekly foot traffic
ax.set_xlabel('Date') # label x-axis
ax.set_ylabel('Sales') # label y-axis

#plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2)) # label ticks on x-axis
fig.autofmt_xdate() # tilt labels on x-axis ticks so they display nicely
plt.tight_layout() # remove whitespace around figure

# 1. Time Series Decomposition
household1_decomp = STL(household1_resample['Sales'], period=12).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(household1_decomp.observed)
ax1.set_ylabel('Observed')

ax2.plot(household1_decomp.trend)
ax2.set_ylabel('Trend')

ax3.plot(household1_decomp.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(household1_decomp.resid)
ax4.set_ylabel('Residuals')

#plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
fig.autofmt_xdate()
plt.tight_layout()

# 2. Test and Transform  Stationarity
ad_fuller_res = adfuller(household1_resample['Sales'])
print(f'ADF Statistic: {ad_fuller_res[0]}')
print(f'p-value: {ad_fuller_res[1]}')

# 3. Correlation Analysis
plot_acf(household1_resample['Sales'], lags=20);
plot_pacf(household1_resample['Sales'], lags=20);

# 1.2. Model Fitting and Selection --------------------------------------------------------------------------

# 1. Setup
#train-test split
train =  household1_resample['Sales'][:60]
test = household1_resample['Sales'].iloc[60:]
exog = household1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']]
exog_train = household1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[:60]
exog_test = household1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[60:]

# 2. Autoregressive Moving Average (ARMA)
    # 2.1. Grid search
ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
result_df = optimize_ARMA(train.astype(float), order_list)
result_df
    # 2.2. Fit optimal parameters
# Choose best model and fit on training data
model = SARIMAX(train.astype(float), order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)
    # 2.3. Residual analysis
# qualitiative analysis
model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = model_fit.resid 
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 3. ARIMA
    # 3.1. Grid Search 
ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 1
order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train.astype(float), order_list, d)
result_df
    # 3.2. Fit optimal parameters
ARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)
    # 3.3. Residual analysis
# qualitative analysis
ARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = ARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 4. SARIMA
    # 4.1. Grid Search
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
    # 4.2. Fit optimal parameters
SARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMA_model_fit = SARIMA_model.fit(disp=False)
    # 4.2. Residual analysis
# qualitative
SARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative
residuals = SARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 5. SARIMAX
    # 5.1. Grid Search
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
SARIMAX_fit.plot_diagnostics(figsize=(10,8));
residuals = SARIMAX_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
lb_result


# 1.3. Forecasts ------------------------------------------------------------------------------------------

# Initialize hyperparameter
target = household1_resample['Sales']
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

pred_mean = rolling_forecast(target, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_ARMA = rolling_forecast(target.astype(float), TRAIN_LEN, HORIZON, WINDOW, 'ARMA')
pred_ARIMA = ARIMA_model_fit.get_prediction(60,64).predicted_mean
pred_SARIMA = SARIMA_model_fit.get_prediction(60, 64).predicted_mean
pred_SARIMAX = rolling_forecast_SARIMA(target.astype(float), exog.astype(float), TRAIN_LEN, HORIZON, WINDOW,'SARIMAX')



pred_df = pd.DataFrame({'actual': test})
pred_df['mean'] = pred_mean
pred_df['ARMA'] = pred_ARMA 
pred_df['ARIMA'] = pred_ARIMA
pred_df['SARIMA'] = pred_SARIMA
pred_df['SARIMAX'] = pred_SARIMAX


from sklearn.metrics import mean_squared_error

error_df = {}

error_df['SARIMAX'] = mean_squared_error(pred_df['actual'], pred_df['SARIMAX'], squared=False)
error_df['ARIMA'] = mean_squared_error(pred_df['actual'], pred_df['ARIMA'], squared=False)
error_df['SARIMA'] = mean_squared_error(pred_df['actual'], pred_df['SARIMA'], squared=False)


# 2. HOUSEHOLD2 ==========================================================
# Resample data
household2_resample = household2
household2_resample['yy-mm']  = household2_resample['Date'].dt.to_period('M')
household2_resample = household2_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
household2_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
household2_resample['Date'] = household2_resample['Date'].dt.to_timestamp()
household2_resample 

# 2.1. Time Series Analysis-------------------------------------------------------
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(household2_resample['Date'], household2_resample['Sales']) # plot avg. weekly foot traffic
ax.set_xlabel('Date') # label x-axis
ax.set_ylabel('Sales') # label y-axis

#plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2)) # label ticks on x-axis
fig.autofmt_xdate() # tilt labels on x-axis ticks so they display nicely
plt.tight_layout() # remove whitespace around figure

# 1. Time Series Decomposition
household2_decomp = STL(household2_resample['Sales'], period=12).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(household1_decomp.observed)
ax1.set_ylabel('Observed')

ax2.plot(household2_decomp.trend)
ax2.set_ylabel('Trend')

ax3.plot(household2_decomp.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(household2_decomp.resid)
ax4.set_ylabel('Residuals')

#plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
fig.autofmt_xdate()
plt.tight_layout()

# 2. Test and Transform  Stationarity
ad_fuller_res = adfuller(household2_resample['Sales'])
print(f'ADF Statistic: {ad_fuller_res[0]}')
print(f'p-value: {ad_fuller_res[1]}')

# 3. Correlation Analysis
plot_acf(household2_resample['Sales'], lags=20);
plot_pacf(household2_resample['Sales'], lags=20);

# 2.2. Model Fitting and Selection --------------------------------------------------------------------------

# 1. Setup
#train-test split
train =  household2_resample['Sales'][:60]
test = household2_resample['Sales'].iloc[60:]
exog = household2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']]
exog_train = household2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[:60]
exog_test = household2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[60:]

# 2. Autoregressive Moving Average (ARMA)
    # 2.1. Grid search
ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
result_df = optimize_ARMA(train.astype(float), order_list)
result_df
    # 2.2. Fit optimal parameters
# Choose best model and fit on training data
model = SARIMAX(train.astype(float), order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)
    # 2.3. Residual analysis
# qualitiative analysis
model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = model_fit.resid 
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 3. ARIMA
    # 3.1. Grid Search 
ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 1
order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train.astype(float), order_list, d)
result_df
    # 3.2. Fit optimal parameters
ARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)
    # 3.3. Residual analysis
# qualitative analysis
ARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = ARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 4. SARIMA
    # 4.1. Grid Search
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
    # 4.2. Fit optimal parameters
SARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMA_model_fit = SARIMA_model.fit(disp=False)
    # 4.2. Residual analysis
# qualitative
SARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative
residuals = SARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 5. SARIMAX
    # 5.1. Grid Search
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
SARIMAX_fit.plot_diagnostics(figsize=(10,8));
residuals = SARIMAX_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
lb_result


# 2.3. Forecasts ------------------------------------------------------------------------------------------

# Initialize hyperparameter
target = household2_resample['Sales']
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

pred_mean = rolling_forecast(target, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_ARMA = rolling_forecast(target.astype(float), TRAIN_LEN, HORIZON, WINDOW, 'ARMA')
pred_ARIMA = ARIMA_model_fit.get_prediction(60,64).predicted_mean
pred_SARIMA = SARIMA_model_fit.get_prediction(60, 64).predicted_mean
pred_SARIMAX = rolling_forecast_SARIMA(target.astype(float), exog.astype(float), TRAIN_LEN, HORIZON, WINDOW,'SARIMAX')

pred_df = pd.DataFrame({'actual': test})
pred_df['mean'] = pred_mean
pred_df['ARMA'] = pred_ARMA 
pred_df['ARIMA'] = pred_ARIMA
pred_df['SARIMA'] = pred_SARIMA
pred_df['SARIMAX'] = pred_SARIMAX

from sklearn.metrics import mean_squared_error
error_df = {}

error_df['SARIMAX'] = mean_squared_error(pred_df['actual'], pred_df['SARIMAX'], squared=False)
error_df['ARIMA'] = mean_squared_error(pred_df['actual'], pred_df['ARIMA'], squared=False)
error_df['SARIMA'] = mean_squared_error(pred_df['actual'], pred_df['SARIMA'], squared=False)

# II.3. HOBBIES ================================================================================================================

# 1. HOBBIES1 ==========================================================
# Resample data
hobbies1_resample = hobbies1
hobbies1_resample['yy-mm']  = hobbies1_resample['Date'].dt.to_period('M')
hobbies1_resample = hobbies1_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
hobbies1_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
hobbies1_resample['Date'] = hobbies1_resample['Date'].dt.to_timestamp()
hobbies1_resample 

# 1.1. Time Series Analysis-------------------------------------------------------
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(hobbies1_resample['Date'], hobbies1_resample['Sales']) # plot avg. weekly foot traffic
ax.set_xlabel('Date') # label x-axis
ax.set_ylabel('Sales') # label y-axis

#plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2)) # label ticks on x-axis
fig.autofmt_xdate() # tilt labels on x-axis ticks so they display nicely
plt.tight_layout() # remove whitespace around figure

# 1. Time Series Decomposition
hobbies1_decomp = STL(hobbies1_resample['Sales'], period=12).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(hobbies1_decomp.observed)
ax1.set_ylabel('Observed')

ax2.plot(hobbies1_decomp.trend)
ax2.set_ylabel('Trend')

ax3.plot(hobbies1_decomp.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(hobbies1_decomp.resid)
ax4.set_ylabel('Residuals')

#plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
fig.autofmt_xdate()
plt.tight_layout()

# 2. Test and Transform  Stationarity
ad_fuller_res = adfuller(hobbies1_resample['Sales'])
print(f'ADF Statistic: {ad_fuller_res[0]}')
print(f'p-value: {ad_fuller_res[1]}')

# 3. Correlation Analysis
plot_acf(hobbies1_resample['Sales'], lags=20);
plot_pacf(hobbies1_resample['Sales'], lags=20);

# 1.2. Model Fitting and Selection --------------------------------------------------------------------------

# 1. Setup
#train-test split
train =  hobbies1_resample['Sales'][:60]
test = hobbies1_resample['Sales'].iloc[60:]
exog = hobbies1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']]
exog_train = hobbies1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[:60]
exog_test = hobbies1_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[60:]

# 2. Autoregressive Moving Average (ARMA)
    # 2.1. Grid search
ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
result_df = optimize_ARMA(train.astype(float), order_list)
result_df
    # 2.2. Fit optimal parameters
# Choose best model and fit on training data
model = SARIMAX(train.astype(float), order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)
    # 2.3. Residual analysis
# qualitiative analysis
model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = model_fit.resid 
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 3. ARIMA
    # 3.1. Grid Search 
ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 1
order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train.astype(float), order_list, d)
result_df
    # 3.2. Fit optimal parameters
ARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)
    # 3.3. Residual analysis
# qualitative analysis
ARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = ARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 4. SARIMA
    # 4.1. Grid Search
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
    # 4.2. Fit optimal parameters
SARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMA_model_fit = SARIMA_model.fit(disp=False)
    # 4.2. Residual analysis
# qualitative
SARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative
residuals = SARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 5. SARIMAX
    # 5.1. Grid Search
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
SARIMAX_fit.plot_diagnostics(figsize=(10,8));
residuals = SARIMAX_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
lb_result


# 1.3. Forecasts ------------------------------------------------------------------------------------------

# Initialize hyperparameter
target = hobbies1_resample['Sales']
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

pred_mean = rolling_forecast(target, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_ARMA = rolling_forecast(target.astype(float), TRAIN_LEN, HORIZON, WINDOW, 'ARMA')
pred_ARIMA = ARIMA_model_fit.get_prediction(60,64).predicted_mean
pred_SARIMA = SARIMA_model_fit.get_prediction(60, 64).predicted_mean
pred_SARIMAX = rolling_forecast_SARIMA(target.astype(float), exog.astype(float), TRAIN_LEN, HORIZON, WINDOW,'SARIMAX')



pred_df = pd.DataFrame({'actual': test})
pred_df['mean'] = pred_mean
pred_df['ARMA'] = pred_ARMA 
pred_df['ARIMA'] = pred_ARIMA
pred_df['SARIMA'] = pred_SARIMA
pred_df['SARIMAX'] = pred_SARIMAX


from sklearn.metrics import mean_squared_error

error_df = {}

error_df['SARIMAX'] = mean_squared_error(pred_df['actual'], pred_df['SARIMAX'], squared=False)
error_df['ARIMA'] = mean_squared_error(pred_df['actual'], pred_df['ARIMA'], squared=False)
error_df['SARIMA'] = mean_squared_error(pred_df['actual'], pred_df['SARIMA'], squared=False)


# 2. HOBBIES2 ==========================================================
# Resample data
hobbies2_resample = hobbies2
hobbies2_resample['yy-mm']  = hobbies2_resample['Date'].dt.to_period('M')
hobbies2_resample = hobbies2_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
hobbies2_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
hobbies2_resample['Date'] = hobbies2_resample['Date'].dt.to_timestamp()
hobbies2_resample 

# 1.1. Time Series Analysis-------------------------------------------------------
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.plot(hobbies2_resample['Date'], hobbies2_resample['Sales']) # plot avg. weekly foot traffic
ax.set_xlabel('Date') # label x-axis
ax.set_ylabel('Sales') # label y-axis

#plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2)) # label ticks on x-axis
fig.autofmt_xdate() # tilt labels on x-axis ticks so they display nicely
plt.tight_layout() # remove whitespace around figure

# 1. Time Series Decomposition
hobbies2_decomp = STL(hobbies2_resample['Sales'], period=12).fit()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(hobbies2_decomp.observed)
ax1.set_ylabel('Observed')

ax2.plot(hobbies2_decomp.trend)
ax2.set_ylabel('Trend')

ax3.plot(hobbies2_decomp.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(hobbies2_decomp.resid)
ax4.set_ylabel('Residuals')

#plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
fig.autofmt_xdate()
plt.tight_layout()

# 2. Test and Transform  Stationarity
ad_fuller_res = adfuller(hobbies2_resample['Sales'])
print(f'ADF Statistic: {ad_fuller_res[0]}')
print(f'p-value: {ad_fuller_res[1]}')

# 3. Correlation Analysis
plot_acf(hobbies2_resample['Sales'], lags=20);
plot_pacf(hobbies2_resample['Sales'], lags=20);

# 2.2. Model Fitting and Selection --------------------------------------------------------------------------

# 1. Setup
#train-test split
train =  hobbies2_resample['Sales'][:60]
test = hobbies2_resample['Sales'].iloc[60:]
exog = hobbies2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']]
exog_train = hobbies2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[:60]
exog_test = hobbies2_resample[['Unemployment', 'Inflation', 'Real Disposable Income']].iloc[60:]

# 2. Autoregressive Moving Average (ARMA)
    # 2.1. Grid search
ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
result_df = optimize_ARMA(train.astype(float), order_list)
result_df
    # 2.2. Fit optimal parameters
# Choose best model and fit on training data
model = SARIMAX(train.astype(float), order=(1,0,1), simple_differencing=False)
model_fit = model.fit(disp=False)
    # 2.3. Residual analysis
# qualitiative analysis
model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = model_fit.resid 
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 3. ARIMA
    # 3.1. Grid Search 
ps = range(0, 4, 1)
qs = range(0, 4, 1)
d = 1
order_list = list(product(ps, qs))
result_df = optimize_ARIMA(train.astype(float), order_list, d)
result_df
    # 3.2. Fit optimal parameters
ARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)
    # 3.3. Residual analysis
# qualitative analysis
ARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative analysis
residuals = ARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 4. SARIMA
    # 4.1. Grid Search
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
    # 4.2. Fit optimal parameters
SARIMA_model = SARIMAX(train.astype(float), order=(1,1,1), seasonal_order=(0, 0, 0, 12), simple_differencing=False)
SARIMA_model_fit = SARIMA_model.fit(disp=False)
    # 4.2. Residual analysis
# qualitative
SARIMA_model_fit.plot_diagnostics(figsize=(10,8));
# quantiative
residuals = SARIMA_model_fit.resid
lb_results = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_results)

# 5. SARIMAX
    # 5.1. Grid Search
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
SARIMAX_fit.plot_diagnostics(figsize=(10,8));
residuals = SARIMAX_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
lb_result


# 1.3. Forecasts ------------------------------------------------------------------------------------------

# Initialize hyperparameter
target = hobbies2_resample['Sales']
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

pred_mean = rolling_forecast(target, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_ARMA = rolling_forecast(target.astype(float), TRAIN_LEN, HORIZON, WINDOW, 'ARMA')
pred_ARIMA = ARIMA_model_fit.get_prediction(60,64).predicted_mean
pred_SARIMA = SARIMA_model_fit.get_prediction(60, 64).predicted_mean
pred_SARIMAX = rolling_forecast_SARIMA(target.astype(float), exog.astype(float), TRAIN_LEN, HORIZON, WINDOW,'SARIMAX')



pred_df = pd.DataFrame({'actual': test})
pred_df['mean'] = pred_mean
pred_df['ARMA'] = pred_ARMA 
pred_df['ARIMA'] = pred_ARIMA
pred_df['SARIMA'] = pred_SARIMA
pred_df['SARIMAX'] = pred_SARIMAX


from sklearn.metrics import mean_squared_error

error_df = {}

error_df['SARIMAX'] = mean_squared_error(pred_df['actual'], pred_df['SARIMAX'], squared=False)
error_df['ARIMA'] = mean_squared_error(pred_df['actual'], pred_df['ARIMA'], squared=False)
error_df['SARIMA'] = mean_squared_error(pred_df['actual'], pred_df['SARIMA'], squared=False)
