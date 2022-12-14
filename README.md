

# Forecasting-Product-Demand

## Project/Goals

The problem we are considering is that of forecasting product demand for a collection of Walmart stores. We are thus faced with a time-series problem. 

## Dataset

The dataset contains the unit sales of various products sold in the USA organizes in a grouped time series. More specifically, the datset contains the unit sales of 30,49 products, classified into three product categories (Hobbies, Foods, and Households) and are in turn disaggregated into 7 product departments per store. The products are sold across 10 stores located in 3 state (CA, TX, WI). 

The bottom level of the grouped data structure is a product – dept – category – store – state combination. Importantly this data structure is not hierarchical because there are multiple aggregation paths, across varying combinations of product and geographical mappings.

[**Aggregation Paths**] Unit Sales of all products can be aggregated and disaggregated by:

- Total (all stores/states)
- State
- Store
- Category
- Department
- Category – State Combination
- Department – State Combination
- Category-Store
- Department Store

Unit sales of some product x can be also aggregated/disaggregated by:
- Store – State Combination
- Store
- State

In other words, we can aggregate/disaggregate along either:

1. Geographic lines
2. Product Lines
3.  Crossing of product and geographic lines. 

A key issue in the project was the size of the dataset; in total, considering all unique product-department-category-store-state-combinations, the number of observations in the dataset was north of 500 million. As a result, the forecasting analysis was conducted on a small sample of data (1%-10% of total) and on seperared and aggregate levels of the data. 

## Models 

The analysis considers the following collection of models:

1. Statistical Models
- ARIMA
- SARIMA
- SARIMAX
2. Neural Network Models
- Linear
- Dense
- LSTM

The analysis contextualizes the models as a series of "progressions"; the statistical models progress in terms of degree of information incorporated in forecast (e.g., seasonality, correleated variables) and the neural netowrk models progress in terms of greater flexibilty in model structure. We hypothesize the more complex models will demonstrate greater accuracy in forecasts for these reasons. 

## Process

The process for generating our collection of forecasts varied between statistical and neural netowrk models. 

### Staistical Forecasts

1. Use problem objective to determine a selection of hyperparameters:
    i. Forecast window
    ii. Forecast horizon
2. Visualize time-series
3. Use time-series for initial tell of seasonality and trend
4. Time-series decomposition to extract trend and seasonal components
5. Use the above steps to determine geenral model to apply (Moving Average, Autoregressive, Autoregressive Moving Average, ARIMA models). 
6. Model time-series:
     i. Apply transformations to make data stationary
     ii. Set values of the difference parameters
     iii. Grid-search over parameters (p,q) to find optimal set of parameters
     iv. Perform residual analysis to validate model
7. Perform rolling forecast of window length *x* over test set of horizon length *y*
8. Compare model performance to actual value, and alternative mdoels by reference to loss function. 

### Neural Network Models

1. Data Preparation
      i. Data exploration
      ii. Train-Validation-Test split
      iii. Feature engineering (Normalizatio/Standardization)
      iv. Variable transformation
2. Define Data Window class and Complile_and_Fit function
3. Define forecast problem as *single-step* or *multi-step* one
4. For a given class of forecasts (baseline, neural network):
      i. Define a dictionary to hold predictions
      ii. Generate forecast window with Data Window class
      iii. Instantiate forecast model, compile and fit model on data window

## Results

An illustration of the resultant forecast performance and be illustrated by way of an example. If we consider a store in California, the forecasts for product sales were off on average by:

- ARIMA: 574 sales
- SARIMA: 574 sales
- SARIMAX: 632 sales
- Dense Neural Net: 0.17 sales
- LSTM Neural Net: 0.07 sales
