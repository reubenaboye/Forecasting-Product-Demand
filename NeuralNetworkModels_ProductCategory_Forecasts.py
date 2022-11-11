################################
# title: LHL Final Project: Neural Network Models - Product Category Forecasts
###############################
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Lambda, Reshape, RNN, LSTMCell, Bidirectional, RNN, Dropout
###################### I. PRELIMINARIES ###############################################
# 1. Import Data------------------------------------------------
# 1.1. Product Data
ProductCat_dta = pd.read_csv('Product_Categories.csv', sep='\t')
ProductCat_dta = ProductCat_dta.drop('Unnamed: 0', axis=1)

ProductCat_hobbies = pd.read_csv('Product_Category_Hobbies.csv', sep='\t')
ProductCat_hobbies = ProductCat_hobbies.drop('Unnamed: 0', axis=1)
ProductCat_foods = pd.read_csv('Product_Category_Foods.csv', sep='\t')
ProductCat_foods = ProductCat_foods.drop('Unnamed: 0', axis=1)
ProductCat_household = pd.read_csv('Product_Category_Household.csv', sep='\t')
ProductCat_household = ProductCat_household.drop('Unnamed: 0', axis=1)

# 2. Function and Classes -------------------------------------------------
class Baseline(Model):
    def __init__ (self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None: # if no target is specified we retrun all columns. 
            return inputs
        elif isinstance(self.label_index, list): # we specofy a list of targets, it returns only the specififed columns
            tensors = []
            for index in self.label_index:
                result = inputs[:, :, index]
                result = result[:, :, tf.newaxis]
                tensors.append(result)
            return tf.concat(tensors, axis=-1)
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]

class MultiStepLastBaseline(Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return tf.tile(inputs[:, -1:, :], [1, 24, 1])
        return tf.tile(inputs[:, -1:, self.label_index:], [1, 24, 1])

def compile_and_fit(model, window, max_epochs=100): 
    model.compile(loss=MeanSquaredError(), # MSE is used as the loss function
    optimizer=Adam(),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]) # MAE is used as an error metric and will be used to compare the performance of our model

    history = model.fit(window.train, # fits model on training ser
    epochs=max_epochs,
    validation_data=window.val) # early_stopping is passed as a callback. 

    return history

#################################### II. PRODUCT CATEGORY FORECASTS ################################################################

# 1. FOODS ==========================================================

# Resample 

foods_resample = ProductCat_foods
foods_resample['yy-mm']  = foods_resample['Date'].dt.to_period('M')
foods_resample = foods_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
foods_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
foods_resample['Date'] = foods_resample['Date'].dt.to_timestamp()
foods_resample 

# Train-Validation-Test Split and Variable Transformation
#train-test split
n = len(foods_resample)
train_df = foods_resample[0:int(n*0.7)]
val_df = foods_resample[int(n*0.7):int(n*0.9)]
test_df = foods_resample[int(n*0.9):]

#Variable Transformation
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_df)
train_df[train_df.columns] = scaler.transform(train_df[train_df.columns])
val_df[val_df.columns] = scaler.transform(val_df[val_df.columns])
test_df[test_df.columns] = scaler.transform(test_df[test_df.columns])

# Instantiate DataWindow to Process Sequence Data
class DataWindow():
    def __init__(self, input_width, label_width, shift, train_df=train_df, val_df=val_df, test_df=test_df, # Initialization function is shown here to : (1) assign the variables and (2) manage indices of the inputs and labels. 
    label_columns=None):
        self.train_df=train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns # name of the column that we wish ti predict
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in # create a dictionary with name and index of the label column. This will be used for plotting
            enumerate(label_columns)} 
            self.column_indices = {name: i for i, name in # Create a dictionary with the name and index of each column. Will be ised to separate the features form target variable
            enumerate(train_df.columns)}
        
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width) # the slice function returns a slice objecy that specifiec how to slice a sequence. In this case it says that the input slice starts at 0 and ends when we reach the input_width
        self.input_indices = np.arange(self.total_window_size)[self.input_slice] # assigns indices to the inputs. This sin uselful for plotting

        self.labels_start = self.total_window_size - self.label_width # get the index at which the label starts. In this case, it is the total window size-width of the label
        self.labels_slice = slice(self.labels_start, None) # the same steps that were applied for the inputs are applied for labels. 
        self.labels_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_to_inputs_labels(self, features): # Here we define a function to help us split our windoes between inputs and labels, allowing models can make predctions based on the inputs and measire an error metric against the labels
        inputs = features[:, self.input_slice, :] # slice the window to the the inputs usif input_slice defined in _init_,
        labels = features[:, self.labels_slice, :] #Slice the window to get the labels using the labels_slice defined in __init__.
        if self.label_columns is not None: # If we have more than one target, we stack the labels
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in 
                self.label_columns],
                axis = -1
            ) 
        inputs.set_shape([None, self.input_width, None]) # The shapew will be [batch, time, features]. Athi this point we specify the time dimensions only and allow batch and feature dimensions to be defined later
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    # KEY: the above function will sepeare the data window into two windows, one for inputs, and the other for the labels. 

    # NEXT: we will define a function to plot the input daya, the preductions, and the actual values. We begin with plotting only three time winots but paramerter cna be changes
    def plot(self, model=None, plot_col='Sales', max_subplots=3):
        inputs, labels = self.sample_batch

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [scaled]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
            label='Inputs', marker='.', zorder=-10)
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
            
            plt.scatter(self.labels_indices, labels[n, :, label_col_index],
            label='Labels', s=64)

            if model is not None:
                predictions=model(inputs)
                plt.scatter(self.labels_indices, predictions [n, :, label_col_index],
                label = 'Predictions', s=64)
            
            if n == 0:
                plt.legend()
        plt.xlabel('Time (h)')

    # LASTLY: we create a function to format our datset into tensors so that they can be fed into the DL models. 
            # We use the tensorflow function 'timeseries_dataset_from_array', which creates a dataset of sliding windoes given an array. 
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size, # Define total length of array, which is equal to total window length
            sequence_stride=1, # define the number of timesteps separating each sequence. In this sace we want sequences to be conseceutive
            shuffle=True, # shuffle sequences. Keep in minf that the data is still in chronological order, we simply are shufflinf the order of the sequences. 
            batch_size=32 # define number of sequences in a batch. 
        )
        ds = ds.map(self.split_to_inputs_labels)
        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    @property
    def val (self):
        return self.make_dataset(self.val_df)
    @property
    def test (self):
        return self.make_dataset(self.test_df)
    @property
    def sample_batch(self):
        result=getattr(self, '_sample_batch', None)
        if result is None:
            result=next(iter(self.train))
            self._sample_batch=result
        return result

# Initialize Window Types
single_step_window = DataWindow(input_width=1, label_width=1, shift=1, label_columns=['Sales'])

wide_window = DataWindow(input_width=4, label_width=4, shift=1, label_columns=['Sales'])

# 1.1. SINGLE-STEP MODELS--------------------------------------------------------------------------

# 1.1.1. Baseline Models
# Generate dictionary with the name and index of each column in the training set
column_indices = {name: i for i, name in enumerate(train_df.columns)}

# Initialize Baseline model
baseline_last = Baseline(label_index=column_indices['Sales']) # Pass index of target column
# Create dictionaries to store forecasts
foods_val_performance = {}
foods_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
foods_val_performance['Baseline - Last'] = baseline_last.evaluate(single_step_window.val)
foods_performance ['Baseline - Last'] = baseline_last.evaluate(single_step_window.test, verbose=0)

# 1.1.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, single_step_window)
foods_val_performance['Linear'] = linear.evaluate(single_step_window.val)
foods_performance['Linear'] = linear.evaluate(single_step_window.test)

# 1.1.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, single_step_window)
foods_val_performance['Dense'] = dense.evaluate(single_step_window.val)
foods_performance['Dense'] = dense.evaluate(single_step_window.test)

# 1.1.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, single_step_window)
foods_val_performance['LSTM'] = lstm_model.evaluate(single_step_window.val)
foods_performance['LSTM'] = lstm_model.evaluate(single_step_window.test)
foods_val_performance
foods_val_performance

# 1.2. MULTI-STEP MODELS -------------------------------------------------------------------------------
# 1.2.1. Baseline Models

# Create dictionaries to store forecasts
ms_foods_val_performance = {}
ms_foods_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
ms_foods_val_performance['Baseline - Last'] = baseline_last.evaluate(wide_window.val)
ms_foods_performance ['Baseline - Last'] = baseline_last.evaluate(wide_window.test, verbose=0)

# 1.2.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, wide_window)
ms_foods_val_performance['Linear'] = linear.evaluate(wide_window.val)
ms_foods_performance['Linear'] = linear.evaluate(wide_window.test)

# 1.2.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, wide_window)
ms_foods_val_performance['Dense'] = dense.evaluate(wide_window.val)
ms_foods_performance['Dense'] = dense.evaluate(wide_window.test)

# 1.3.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, wide_window)
ms_foods_val_performance['LSTM'] = lstm_model.evaluate(wide_window)
ms_foods_performance['LSTM'] = lstm_model.evaluate(wide_window.test)
ms_foods_val_performance
ms_foods_val_performance


# 2. HOUSEHOLD =================================================================================================

# Resample 
household_resample = ProductCat_household
household_resample['yy-mm']  = household_resample['Date'].dt.to_period('M')
foods_resample = foods_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
foods_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
foods_resample['Date'] = foods_resample['Date'].dt.to_timestamp()
foods_resample 

# Train-Validation-Test Split and Variable Transformation
#train-test split
n = len(household_resample)
train_df = household_resample[0:int(n*0.7)]
val_df = household_resample[int(n*0.7):int(n*0.9)]
test_df = household_resample[int(n*0.9):]

#Variable Transformation
scaler.fit(train_df)
train_df[train_df.columns] = scaler.transform(train_df[train_df.columns])
val_df[val_df.columns] = scaler.transform(val_df[val_df.columns])
test_df[test_df.columns] = scaler.transform(test_df[test_df.columns])

# Instantiate DataWindow to Process Sequence Data
class DataWindow():
    def __init__(self, input_width, label_width, shift, train_df=train_df, val_df=val_df, test_df=test_df, # Initialization function is shown here to : (1) assign the variables and (2) manage indices of the inputs and labels. 
    label_columns=None):
        self.train_df=train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns # name of the column that we wish ti predict
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in # create a dictionary with name and index of the label column. This will be used for plotting
            enumerate(label_columns)} 
            self.column_indices = {name: i for i, name in # Create a dictionary with the name and index of each column. Will be ised to separate the features form target variable
            enumerate(train_df.columns)}
        
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width) # the slice function returns a slice objecy that specifiec how to slice a sequence. In this case it says that the input slice starts at 0 and ends when we reach the input_width
        self.input_indices = np.arange(self.total_window_size)[self.input_slice] # assigns indices to the inputs. This sin uselful for plotting

        self.labels_start = self.total_window_size - self.label_width # get the index at which the label starts. In this case, it is the total window size-width of the label
        self.labels_slice = slice(self.labels_start, None) # the same steps that were applied for the inputs are applied for labels. 
        self.labels_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_to_inputs_labels(self, features): # Here we define a function to help us split our windoes between inputs and labels, allowing models can make predctions based on the inputs and measire an error metric against the labels
        inputs = features[:, self.input_slice, :] # slice the window to the the inputs usif input_slice defined in _init_,
        labels = features[:, self.labels_slice, :] #Slice the window to get the labels using the labels_slice defined in __init__.
        if self.label_columns is not None: # If we have more than one target, we stack the labels
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in 
                self.label_columns],
                axis = -1
            ) 
        inputs.set_shape([None, self.input_width, None]) # The shapew will be [batch, time, features]. Athi this point we specify the time dimensions only and allow batch and feature dimensions to be defined later
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    # KEY: the above function will sepeare the data window into two windows, one for inputs, and the other for the labels. 

    # NEXT: we will define a function to plot the input daya, the preductions, and the actual values. We begin with plotting only three time winots but paramerter cna be changes
    def plot(self, model=None, plot_col='Sales', max_subplots=3):
        inputs, labels = self.sample_batch

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [scaled]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
            label='Inputs', marker='.', zorder=-10)
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
            
            plt.scatter(self.labels_indices, labels[n, :, label_col_index],
            label='Labels', s=64)

            if model is not None:
                predictions=model(inputs)
                plt.scatter(self.labels_indices, predictions [n, :, label_col_index],
                label = 'Predictions', s=64)
            
            if n == 0:
                plt.legend()
        plt.xlabel('Time (h)')

    # LASTLY: we create a function to format our datset into tensors so that they can be fed into the DL models. 
            # We use the tensorflow function 'timeseries_dataset_from_array', which creates a dataset of sliding windoes given an array. 
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size, # Define total length of array, which is equal to total window length
            sequence_stride=1, # define the number of timesteps separating each sequence. In this sace we want sequences to be conseceutive
            shuffle=True, # shuffle sequences. Keep in minf that the data is still in chronological order, we simply are shufflinf the order of the sequences. 
            batch_size=32 # define number of sequences in a batch. 
        )
        ds = ds.map(self.split_to_inputs_labels)
        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    @property
    def val (self):
        return self.make_dataset(self.val_df)
    @property
    def test (self):
        return self.make_dataset(self.test_df)
    @property
    def sample_batch(self):
        result=getattr(self, '_sample_batch', None)
        if result is None:
            result=next(iter(self.train))
            self._sample_batch=result
        return result

# Initialize Window Types
single_step_window = DataWindow(input_width=1, label_width=1, shift=1, label_columns=['Sales'])

wide_window = DataWindow(input_width=4, label_width=4, shift=1, label_columns=['Sales'])

# 2.1. SINGLE-STEP MODELS------------------------------------------------------------------

# 2.1.1. Baseline Models
# Generate dictionary with the name and index of each column in the training set
column_indices = {name: i for i, name in enumerate(train_df.columns)}

# Initialize Baseline model
baseline_last = Baseline(label_index=column_indices['Sales']) # Pass index of target column
# Create dictionaries to store forecasts
household_val_performance = {}
household_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
household_val_performance['Baseline - Last'] = baseline_last.evaluate(single_step_window.val)
household_performance ['Baseline - Last'] = baseline_last.evaluate(single_step_window.test, verbose=0)

# 2.1.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, single_step_window)
foods_val_performance['Linear'] = linear.evaluate(single_step_window.val)
foods_performance['Linear'] = linear.evaluate(single_step_window.test)

# 2.1.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, single_step_window)
household_val_performance['Dense'] = dense.evaluate(single_step_window.val)
household_performance['Dense'] = dense.evaluate(single_step_window.test)

# 2.1.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, single_step_window)
household_val_performance['LSTM'] = lstm_model.evaluate(single_step_window.val)
household_performance['LSTM'] = lstm_model.evaluate(single_step_window.test)
household_val_performance
household_val_performance

# 2.2 MULTI-STEP MODELS -------------------------------------------------------------------------------
# 2.2.1. Baseline Models

# Create dictionaries to store forecasts
ms_household_val_performance = {}
ms_household_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
ms_household_val_performance['Baseline - Last'] = baseline_last.evaluate(wide_window.val)
ms_household_performance ['Baseline - Last'] = baseline_last.evaluate(wide_window.test, verbose=0)

# 2.2.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, wide_window)
ms_household_val_performance['Linear'] = linear.evaluate(wide_window.val)
ms_household_performance['Linear'] = linear.evaluate(wide_window.test)

# 2.2.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, wide_window)
ms_household_val_performance['Dense'] = dense.evaluate(wide_window.val)
ms_household_performance['Dense'] = dense.evaluate(wide_window.test)

# 2.3.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, wide_window)
ms_household_val_performance['LSTM'] = lstm_model.evaluate(wide_window)
ms_household_performance['LSTM'] = lstm_model.evaluate(wide_window.test)
ms_household_val_performance
ms_household_performance

# 3. HOBBIES =================================================================================================

# Resample 
hobbies_resample = ProductCat_hobbies
hobbies_resample['yy-mm']  = hobbies_resample['Date'].dt.to_period('M')
hobbies_resample = hobbies_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
hobbies_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
hobbies_resample['Date'] = foods_resample['Date'].dt.to_timestamp()
hobbies_resample 

# Train-Validation-Test Split and Variable Transformation
#train-test split
n = len(household_resample)
train_df = hobbies_resample[0:int(n*0.7)]
val_df = hobbies_resample[int(n*0.7):int(n*0.9)]
test_df = hobbies_resample[int(n*0.9):]

#Variable Transformation
scaler.fit(train_df)
train_df[train_df.columns] = scaler.transform(train_df[train_df.columns])
val_df[val_df.columns] = scaler.transform(val_df[val_df.columns])
test_df[test_df.columns] = scaler.transform(test_df[test_df.columns])

# Instantiate DataWindow to Process Sequence Data
class DataWindow():
    def __init__(self, input_width, label_width, shift, train_df=train_df, val_df=val_df, test_df=test_df, # Initialization function is shown here to : (1) assign the variables and (2) manage indices of the inputs and labels. 
    label_columns=None):
        self.train_df=train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns # name of the column that we wish ti predict
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in # create a dictionary with name and index of the label column. This will be used for plotting
            enumerate(label_columns)} 
            self.column_indices = {name: i for i, name in # Create a dictionary with the name and index of each column. Will be ised to separate the features form target variable
            enumerate(train_df.columns)}
        
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width) # the slice function returns a slice objecy that specifiec how to slice a sequence. In this case it says that the input slice starts at 0 and ends when we reach the input_width
        self.input_indices = np.arange(self.total_window_size)[self.input_slice] # assigns indices to the inputs. This sin uselful for plotting

        self.labels_start = self.total_window_size - self.label_width # get the index at which the label starts. In this case, it is the total window size-width of the label
        self.labels_slice = slice(self.labels_start, None) # the same steps that were applied for the inputs are applied for labels. 
        self.labels_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_to_inputs_labels(self, features): # Here we define a function to help us split our windoes between inputs and labels, allowing models can make predctions based on the inputs and measire an error metric against the labels
        inputs = features[:, self.input_slice, :] # slice the window to the the inputs usif input_slice defined in _init_,
        labels = features[:, self.labels_slice, :] #Slice the window to get the labels using the labels_slice defined in __init__.
        if self.label_columns is not None: # If we have more than one target, we stack the labels
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in 
                self.label_columns],
                axis = -1
            ) 
        inputs.set_shape([None, self.input_width, None]) # The shapew will be [batch, time, features]. Athi this point we specify the time dimensions only and allow batch and feature dimensions to be defined later
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    # KEY: the above function will sepeare the data window into two windows, one for inputs, and the other for the labels. 

    # NEXT: we will define a function to plot the input daya, the preductions, and the actual values. We begin with plotting only three time winots but paramerter cna be changes
    def plot(self, model=None, plot_col='Sales', max_subplots=3):
        inputs, labels = self.sample_batch

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [scaled]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
            label='Inputs', marker='.', zorder=-10)
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
            
            plt.scatter(self.labels_indices, labels[n, :, label_col_index],
            label='Labels', s=64)

            if model is not None:
                predictions=model(inputs)
                plt.scatter(self.labels_indices, predictions [n, :, label_col_index],
                label = 'Predictions', s=64)
            
            if n == 0:
                plt.legend()
        plt.xlabel('Time (h)')

    # LASTLY: we create a function to format our datset into tensors so that they can be fed into the DL models. 
            # We use the tensorflow function 'timeseries_dataset_from_array', which creates a dataset of sliding windoes given an array. 
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size, # Define total length of array, which is equal to total window length
            sequence_stride=1, # define the number of timesteps separating each sequence. In this sace we want sequences to be conseceutive
            shuffle=True, # shuffle sequences. Keep in minf that the data is still in chronological order, we simply are shufflinf the order of the sequences. 
            batch_size=32 # define number of sequences in a batch. 
        )
        ds = ds.map(self.split_to_inputs_labels)
        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    @property
    def val (self):
        return self.make_dataset(self.val_df)
    @property
    def test (self):
        return self.make_dataset(self.test_df)
    @property
    def sample_batch(self):
        result=getattr(self, '_sample_batch', None)
        if result is None:
            result=next(iter(self.train))
            self._sample_batch=result
        return result

# Initialize Window Types
single_step_window = DataWindow(input_width=1, label_width=1, shift=1, label_columns=['Sales'])

wide_window = DataWindow(input_width=4, label_width=4, shift=1, label_columns=['Sales'])

# 2.1. SINGLE-STEP MODELS------------------------------------------------------------------

# 2.1.1. Baseline Models
# Generate dictionary with the name and index of each column in the training set
column_indices = {name: i for i, name in enumerate(train_df.columns)}

# Initialize Baseline model
baseline_last = Baseline(label_index=column_indices['Sales']) # Pass index of target column
# Create dictionaries to store forecasts
hobbies_val_performance = {}
hobbies_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
hobbies_val_performance['Baseline - Last'] = baseline_last.evaluate(single_step_window.val)
hobbies_performance ['Baseline - Last'] = baseline_last.evaluate(single_step_window.test, verbose=0)

# 2.1.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, single_step_window)
hobbies_val_performance['Linear'] = linear.evaluate(single_step_window.val)
hobbies_performance['Linear'] = linear.evaluate(single_step_window.test)

# 2.1.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, single_step_window)
hobbies_val_performance['Dense'] = dense.evaluate(single_step_window.val)
hobbies_performance['Dense'] = dense.evaluate(single_step_window.test)

# 2.1.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, single_step_window)
hobbies_val_performance['LSTM'] = lstm_model.evaluate(single_step_window.val)
hobbies_performance['LSTM'] = lstm_model.evaluate(single_step_window.test)
hobbies_val_performance
hobbies_val_performance

# 2.2 MULTI-STEP MODELS -------------------------------------------------------------------------------
# 2.2.1. Baseline Models

# Create dictionaries to store forecasts
ms_hobbies_val_performance = {}
ms_hobbies_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
ms_hobbies_val_performance['Baseline - Last'] = baseline_last.evaluate(wide_window.val)
ms_hobbies_performance ['Baseline - Last'] = baseline_last.evaluate(wide_window.test, verbose=0)

# 2.2.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, wide_window)
ms_hobbies_val_performance['Linear'] = linear.evaluate(wide_window.val)
ms_hobbies_performance['Linear'] = linear.evaluate(wide_window.test)

# 2.2.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, wide_window)
ms_hobbies_val_performance['Dense'] = dense.evaluate(wide_window.val)
ms_hobbies_performance['Dense'] = dense.evaluate(wide_window.test)

# 2.3.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, wide_window)
ms_hobbies_val_performance['LSTM'] = lstm_model.evaluate(wide_window)
ms_hobbies_performance['LSTM'] = lstm_model.evaluate(wide_window.test)
ms_hobbies_val_performance
ms_hobbies_performance





