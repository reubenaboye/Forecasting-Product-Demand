######################
# LHL Final Project: Data Preparation
########################
from calendar import calendar
import numpy as np
import pandas as pd
import os
############################# I. Import and Clean #######################################################
calendar_df = pd.read_csv('calendar[1].csv', parse_dates=['date'])
sales = pd.read_csv('sales_train_evaluation.csv')
price_dta = pd.read_csv('sell_prices.csv')
econ_var = pd.read_csv('econ_dta(2011-2017).csv')

# 1. Create Keys from Dataframe
sales_T = sales.melt(id_vars=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
sales_T

# 2. Take sample from data and rename variables
sample_sales = sales_T.sample(frac=0.1, replace=True, random_state=1)
    # NOTE: This Sample is done so that it can run on my locacl computer. For Neural Networks Models, sample should be larger to get more reflective results
sample_sales.rename(columns={'value': 'sales_qty'}, inplace=True)
sample_sales 

# 3. Clean and Merge Dataframes to Sales data

# 3.1. Calendar Data 
calendar_df = calendar_df.loc[:, ['date', 'wm_yr_wk', 'd', 'month', 'year', 'wday']]
sales_df = sample_sales.merge(calendar_df, left_on='variable', right_on='d', how='left')
sales_df

# 3.2. Price Data
sales_df = pd.merge(sales_df, price_dta, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
sales_df

# 3.3. Economic Data
econ_var = econ_var.drop(['Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13'], axis=1)
econ_var.rename(columns={'Year': 'year', 'Month': 'month'}, inplace=True)
sales_df = pd.merge(sales_df, econ_var, how='left', on=['month', 'year'])
sales_df

# sort by date
sales_df = sales_df.sort_values('date')
sales_df

# 3.4. Check for null values
def null_values_perc(df):
    total_null = df.isnull().sum().sort_values(ascending=False)
    percent_null = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False) 
    null_df = pd.concat([total_null, percent_null], axis=1, keys=['Total', 'Percent'])
    return null_df
null_values_perc(sales_df)
# Fill null values in Price col
def fill_null_val_mean(df, df_col):
    """
    Fills null values in a column with the column's mean value. 
    Note: dataframe column (df_col) must be of numeric type
    """
    col_mean = df[df_col].mean()
    df[df_col].fillna(col_mean, inplace=True)
    return print("Number of Nulls left:",df[df_col].isnull().sum())

fill_null_val_mean(sales_df, 'sell_price')

######################### II. Group and Aggregate Data ################################################

# 1. Geographic --------------------------------------------------
    # 1.1. Store Level
stores_level_dta = sales_df.groupby(['date','store_id', 'month', 'year', 'Unemployment Rate', 'Inflation (1-Month % Change)', 'Real_Disposable_Income (in Billions)'])[['sales_qty']].sum()
stores_level_dta.reset_index(inplace=True)
stores_level_dta = stores_level_dta.T.reset_index(drop=True).T
stores_level_dta.columns = ['Date','Store', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real_Disposable_Income','Sales']
stores_level_dta
        # 1.1.1. CA Stores
CA_1_dta = stores_level_dta[stores_level_dta['Store']=='CA_1']
CA_1_dta.reset_index(inplace=True)
CA_1_dta = CA_1_dta.T.reset_index(drop=True).T
CA_1_dta = CA_1_dta.drop(0, axis=1)
CA_1_dta.columns = ['Date', 'Store', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
CA_1_dta

CA_2_dta = stores_level_dta[stores_level_dta['Store']=='CA_2']
CA_2_dta.reset_index(inplace=True)
CA_2_dta = CA_2_dta.T.reset_index(drop=True).T
CA_2_dta = CA_2_dta.drop(0, axis=1)
CA_2_dta.columns = ['Date', 'Store', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
CA_2_dta

CA_3_dta = stores_level_dta[stores_level_dta['Store']=='CA_3']
CA_3_dta.reset_index(inplace=True)
CA_3_dta = CA_3_dta.T.reset_index(drop=True).T
CA_3_dta = CA_3_dta.drop(0, axis=1)
CA_3_dta.columns = ['Date', 'Store', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
CA_3_dta
        # 1.1.2. TX Stores
TX_1_dta = stores_level_dta[stores_level_dta['Store']=='TX_1']
TX_1_dta.reset_index(inplace=True)
TX_1_dta = TX_1_dta.T.reset_index(drop=True).T
TX_1_dta = TX_1_dta.drop(0, axis=1)
TX_1_dta.columns = ['Date', 'Store', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
TX_1_dta

TX_2_dta = stores_level_dta[stores_level_dta['Store']=='TX_2']
TX_2_dta.reset_index(inplace=True)
TX_2_dta = TX_2_dta.T.reset_index(drop=True).T
TX_2_dta = TX_2_dta.drop(0, axis=1)
TX_2_dta.columns = ['Date', 'Store', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
TX_2_dta

TX_3_dta = stores_level_dta[stores_level_dta['Store']=='TX_3']
TX_3_dta.reset_index(inplace=True)
TX_3_dta = TX_3_dta.T.reset_index(drop=True).T
TX_3_dta = TX_3_dta.drop(0, axis=1)
TX_3_dta.columns = ['Date', 'Store', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
TX_3_dta
        # 1.1.3. WI Stores
WI_1_dta = stores_level_dta[stores_level_dta['Store']=='WI_1']
WI_1_dta.reset_index(inplace=True)
WI_1_dta = WI_1_dta.T.reset_index(drop=True).T
WI_1_dta = WI_1_dta.drop(0, axis=1)
WI_1_dta.columns = ['Date', 'Store', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
WI_1_dta

WI_2_dta = stores_level_dta[stores_level_dta['Store']=='WI_2']
WI_2_dta.reset_index(inplace=True)
WI_2_dta = WI_2_dta.T.reset_index(drop=True).T
WI_2_dta = WI_2_dta.drop(0, axis=1)
WI_2_dta.columns = ['Date', 'Store', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
WI_2_dta

WI_3_dta = stores_level_dta[stores_level_dta['Store']=='WI_3']
WI_3_dta.reset_index(inplace=True)
WI_3_dta = WI_3_dta.T.reset_index(drop=True).T
WI_3_dta = WI_3_dta.drop(0, axis=1)
WI_3_dta.columns = ['Date', 'Store', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
WI_3_dta

    # 1.2. State Level
state_level_dta = sales_df.groupby(['date','state_id', 'month', 'year', 'Unemployment Rate', 'Inflation (1-Month % Change)', 'Real_Disposable_Income (in Billions)'])[['sales_qty']].sum()
state_level_dta.reset_index(inplace=True)
state_level_dta= state_level_dta.T.reset_index(drop=True).T
state_level_dta.columns = ['Date', 'State', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
state_level_dta

        # 1.2.1. CA
CA_dta = state_level_dta[state_level_dta['State']=='CA']
CA_dta.reset_index(inplace=True)
CA_dta = CA_dta.T.reset_index(drop=True).T
CA_dta = CA_dta.drop(0, axis=1)
CA_dta.columns = ['Date', 'State', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
CA_dta
        # 1.2.2. TX
TX_dta = state_level_dta[state_level_dta['State']=='TX']
TX_dta.reset_index(inplace=True)
TX_dta = TX_dta.T.reset_index(drop=True).T
TX_dta = TX_dta.drop(0, axis=1)
TX_dta.columns = ['Date', 'State', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
TX_dta
        # 1.2.3. WI
WI_dta = state_level_dta[state_level_dta['State']=='WI']
WI_dta.reset_index(inplace=True)
WI_dta = WI_dta.T.reset_index(drop=True).T
WI_dta = WI_dta.drop(0, axis=1)
WI_dta.columns = ['Date', 'State', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
WI_dta

# 2. Product ------------------------------------------------------------
    # 2.1. Product Categories
prod_cat_df = sales_df.groupby(['date','cat_id', 'month', 'year', 'Unemployment Rate', 'Inflation (1-Month % Change)', 'Real_Disposable_Income (in Billions)'])[['sales_qty']].sum()
prod_cat_df.reset_index(inplace=True)
prod_cat_df = prod_cat_df.T.reset_index(drop=True).T
prod_cat_df.columns = ['Date', 'Product Category', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
prod_cat_df
        # 2.1.1. Hobbies
hobbies_df = prod_cat_df[prod_cat_df['Product Category']=='HOBBIES']
hobbies_df.reset_index(inplace=True)
hobbies_df= hobbies_df.T.reset_index(drop=True).T
hobbies_df = hobbies_df.drop(0, axis=1)
hobbies_df.columns = ['Date', 'Product Category', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
hobbies_df
        # 2.1.2. Food
food_df = prod_cat_df[prod_cat_df['Product Category']=='FOODS']
food_df.reset_index(inplace=True)
food_df= food_df.T.reset_index(drop=True).T
food_df = food_df.drop(0, axis=1)
food_df.columns = ['Date', 'Product Category', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
food_df
        # 2.1.3. Household
household_df = prod_cat_df[prod_cat_df['Product Category']=='HOUSEHOLD']
household_df.reset_index(inplace=True)
household_df= household_df.T.reset_index(drop=True).T
household_df = household_df.drop(0, axis=1)
household_df.columns = ['Date', 'Product Category', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
household_df

    # 2.2. Product Dept
prod_dept_df = sales_df.groupby(['date','dept_id', 'month', 'year', 'Unemployment Rate', 'Inflation (1-Month % Change)', 'Real_Disposable_Income (in Billions)'])[['sales_qty']].sum()
prod_dept_df.reset_index(inplace=True)
prod_dept_df = prod_dept_df.T.reset_index(drop=True).T
prod_dept_df.columns = ['Date', 'Product Dept', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
prod_dept_df
        # 2.2.1. Food Departments
foods1_df = prod_dept_df[prod_dept_df['Product Dept']=='FOODS_1']
foods1_df.reset_index(inplace=True)
foods1_df = foods1_df.drop('index', axis=1)
foods1_df.columns = ['Date', 'Product Dept', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
foods1_df 

foods2_df = prod_dept_df[prod_dept_df['Product Dept']=='FOODS_2']
foods2_df.reset_index(inplace=True)
foods2_df = foods2_df.drop('index', axis=1)
foods2_df.columns = ['Date', 'Product Dept', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
foods2_df

foods3_df = prod_dept_df[prod_dept_df['Product Dept']=='FOODS_3']
foods3_df.reset_index(inplace=True)
foods3_df = foods3_df.drop('index', axis=1)
foods1_df.columns = ['Date', 'Product Dept', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
foods3_df
        # 2.2.2. Hobbies Departments
hobbies1_df = prod_dept_df[prod_dept_df['Product Dept']=='HOBBIES_1']
hobbies1_df.reset_index(inplace=True)
hobbies1_df = hobbies1_df.drop('index', axis=1)
hobbies1_df.columns = ['Date', 'Product Dept', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
hobbies1_df

hobbies2_df = prod_dept_df[prod_dept_df['Product Dept']=='HOBBIES_2']
hobbies2_df.reset_index(inplace=True)
hobbies2_df = hobbies2_df.drop('index', axis=1)
hobbies2_df.columns = ['Date', 'Product Dept', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
hobbies2_df

        # 2.2.3. Household Departments
household1_df = prod_dept_df[prod_dept_df['Product Dept']=='HOUSEHOLD_1']
household1_df.reset_index(inplace=True)
household1_df = household1_df.drop('index', axis=1)
household1_df.columns = ['Date', 'Product Dept', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
household1_df

household2_df = prod_dept_df[prod_dept_df['Product Dept']=='HOUSEHOLD_2']
household2_df.reset_index(inplace=True)
household2_df = household2_df.drop('index', axis=1)
household2_df.columns = ['Date', 'Product Dept', 'Month', 'Year', 'Unemployment', 'Inflation', 'Real Disposable Income','Sales']
household2_df

######################### III. Export Aggregated and Grouped Daat ############################

stores_level_dta.to_csv('Store_Data.csv', sep='\t')
CA_1_dta.to_csv('CA_Store1.csv', sep='\t')
CA_2_dta.to_csv('CA_Store2.csv', sep='\t')
CA_3_dta.to_csv('CA_Store3.csv', sep='\t')
CA_dta.to_csv('CA_State.csv', sep='\t')
TX_dta.to_csv('TX_State.csv', sep='\t')
WI_dta.to_csv('WI_State.csv', sep='\t')
TX_1_dta.to_csv('TX_Store1.csv', sep='\t')
TX_2_dta.to_csv('TX_Store2.csv', sep='\t')
TX_3_dta.to_csv('TX_Store3.csv', sep='\t')
WI_1_dta.to_csv('WI_Store1.csv', sep='\t')
WI_2_dta.to_csv('WI_Store2.csv', sep='\t')
WI_3_dta.to_csv('WI_Store3.csv', sep='\t')

prod_cat_df.to_csv('Product_Categories.csv', sep='\t')
hobbies_df.to_csv('Product_Category_Hobbies.csv', sep='\t')
food_df.to_csv('Product_Category_Foods.csv', sep='\t')
household_df.to_csv('Product_Category_Household.csv', sep='\t')

prod_dept_df.to_csv('Product_Departments.csv', sep='\t')
foods1_df.to_csv('Prod_Dept_Foods1.csv', sep='\t')
foods2_df.to_csv('Prod_Dept_Foods2.csv', sep='\t')
foods3_df.to_csv('Prod_Dept_Foods3.csv', sep='\t')

hobbies1_df.to_csv('Prod_Dept_Hobbies1.csv', sep='\t')
hobbies2_df.to_csv('Prod_Dept_Hobbies2.csv', sep='\t')
household1_df.to_csv('Prod_Dept_Household1.csv', sep='\t')
household2_df.to_csv('Prod_Dept_Household2.csv', sep='\t')






