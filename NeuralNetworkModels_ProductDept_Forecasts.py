################################
# title: LHL Final Project: Neural Network Models - Product Department Forecasts
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

ProductDept_dta = pd.read_csv('Product_Departments.csv', sep='\t')
ProductDept_dta = ProductDept_dta.drop('Unnamed: 0', axis=1)

ProductDept_foods1 = pd.read_csv('Prod_Dept_Foods1.csv', sep='\t')
ProductDept_foods1 = ProductDept_foods1.drop('Unnamed: 0', axis=1)
ProductDept_foods2 =  pd.read_csv('Product_Dept_Foods2.csv', sep='\t')
ProductDept_foods2 = ProductDept_foods2.drop('Unnamed: 0', axis=1)
ProductDept_foods3 =  pd.read_csv('Product_Dept_Foods3.csv', sep='\t')
ProductDept_foods3 = ProductDept_foods3.drop('Unnamed: 0', axis=1)

ProductDept_hobbies1 =  pd.read_csv('Product_Dept_Hobbies1.csv', sep='\t')
ProductDept_hobbies1 = ProductDept_hobbies1.drop('Unnamed: 0', axis=1)
ProductDept_hobbies2 =  pd.read_csv('Product_Dept_Hobbies2.csv', sep='\t')
ProductDept_hobbies2 = ProductDept_hobbies2.drop('Unnamed: 0', axis=1)

ProductDept_household1 =  pd.read_csv('Product_Dept_Household1.csv', sep='\t')
ProductDept_household1 = ProductDept_household1.drop('Unnamed: 0', axis=1)
ProductDept_household2 =  pd.read_csv('Product_Dept_Household2.csv', sep='\t')
ProductDept_household2 = ProductDept_household2.drop('Unnamed: 0', axis=1)

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

#################################### II. PRODUCT DEPARTMENT FORECASTS ################################################################

# II.1. FOODS ===========================================================================================================================

# 1. FOODS1 ------------------------------------------------------------------------------------------------
# Resample 
foods1_resample = ProductDept_foods1
foods1_resample['yy-mm']  = foods1_resample['Date'].dt.to_period('M')
foods1_resample = foods1_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
foods1_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
foods1_resample['Date'] = foods1_resample['Date'].dt.to_timestamp()
foods1_resample 

# Train-Validation-Test Split and Variable Transformation
#train-test split
n = len(foods1_resample)
train_df = foods1_resample[0:int(n*0.7)]
val_df = foods1_resample[int(n*0.7):int(n*0.9)]
test_df = foods1_resample[int(n*0.9):]

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
foods1_val_performance = {}
foods1_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
foods1_val_performance['Baseline - Last'] = baseline_last.evaluate(single_step_window.val)
foods1_performance ['Baseline - Last'] = baseline_last.evaluate(single_step_window.test, verbose=0)

# 1.1.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, single_step_window)
foods1_val_performance['Linear'] = linear.evaluate(single_step_window.val)
foods1_performance['Linear'] = linear.evaluate(single_step_window.test)

# 1.1.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, single_step_window)
foods1_val_performance['Dense'] = dense.evaluate(single_step_window.val)
foods1_performance['Dense'] = dense.evaluate(single_step_window.test)

# 1.1.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, single_step_window)
foods1_val_performance['LSTM'] = lstm_model.evaluate(single_step_window.val)
foods1_performance['LSTM'] = lstm_model.evaluate(single_step_window.test)
foods1_val_performance
foods1_val_performance

# 1.2. MULTI-STEP MODELS -------------------------------------------------------------------------------
# 1.2.1. Baseline Models

# Create dictionaries to store forecasts
ms_foods1_val_performance = {}
ms_foods1_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
ms_foods1_val_performance['Baseline - Last'] = baseline_last.evaluate(wide_window.val)
ms_foods1_performance ['Baseline - Last'] = baseline_last.evaluate(wide_window.test, verbose=0)

# 1.2.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, wide_window)
ms_foods1_val_performance['Linear'] = linear.evaluate(wide_window.val)
ms_foods1_performance['Linear'] = linear.evaluate(wide_window.test)

# 1.2.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, wide_window)
ms_foods1_val_performance['Dense'] = dense.evaluate(wide_window.val)
ms_foods1_performance['Dense'] = dense.evaluate(wide_window.test)

# 1.3.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, wide_window)
ms_foods1_val_performance['LSTM'] = lstm_model.evaluate(wide_window)
ms_foods1_performance['LSTM'] = lstm_model.evaluate(wide_window.test)
ms_foods1_val_performance
ms_foods1_val_performance

# 2. FOODS2 ----------------------------------------------------------------------------------------------------------------------------

foods2_resample = ProductDept_foods2
foods2_resample['yy-mm']  = foods2_resample['Date'].dt.to_period('M')
foods2_resample = foods2_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
foods2_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
foods2_resample['Date'] = foods2_resample['Date'].dt.to_timestamp()
foods2_resample 

# Train-Validation-Test Split and Variable Transformation
#train-test split
n = len(foods1_resample)
train_df = foods2_resample[0:int(n*0.7)]
val_df = foods2_resample[int(n*0.7):int(n*0.9)]
test_df = foods2_resample[int(n*0.9):]

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
foods2_val_performance = {}
foods2_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
foods2_val_performance['Baseline - Last'] = baseline_last.evaluate(single_step_window.val)
foods2_performance ['Baseline - Last'] = baseline_last.evaluate(single_step_window.test, verbose=0)

# 1.1.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, single_step_window)
foods2_val_performance['Linear'] = linear.evaluate(single_step_window.val)
foods2_performance['Linear'] = linear.evaluate(single_step_window.test)

# 1.1.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, single_step_window)
foods2_val_performance['Dense'] = dense.evaluate(single_step_window.val)
foods2_performance['Dense'] = dense.evaluate(single_step_window.test)

# 1.1.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, single_step_window)
foods2_val_performance['LSTM'] = lstm_model.evaluate(single_step_window.val)
foods2_performance['LSTM'] = lstm_model.evaluate(single_step_window.test)
foods2_val_performance
foods2_val_performance

# 1.2. MULTI-STEP MODELS -------------------------------------------------------------------------------
# 1.2.1. Baseline Models

# Create dictionaries to store forecasts
ms_foods2_val_performance = {}
ms_foods2_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
ms_foods2_val_performance['Baseline - Last'] = baseline_last.evaluate(wide_window.val)
ms_foods2_performance ['Baseline - Last'] = baseline_last.evaluate(wide_window.test, verbose=0)

# 1.2.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, wide_window)
ms_foods2_val_performance['Linear'] = linear.evaluate(wide_window.val)
ms_foods2_performance['Linear'] = linear.evaluate(wide_window.test)

# 1.2.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, wide_window)
ms_foods2_val_performance['Dense'] = dense.evaluate(wide_window.val)
ms_foods2_performance['Dense'] = dense.evaluate(wide_window.test)

# 1.3.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, wide_window)
ms_foods2_val_performance['LSTM'] = lstm_model.evaluate(wide_window)
ms_foods2_performance['LSTM'] = lstm_model.evaluate(wide_window.test)
ms_foods2_val_performance
ms_foods2_val_performance

# 3. FOODS3 -------------------------------------------------------------------------------------------------------------

foods3_resample = ProductDept_foods1
foods3_resample['yy-mm']  = foods3_resample['Date'].dt.to_period('M')
foods3_resample = foods3_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
foods3_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
foods3_resample['Date'] = foods3_resample['Date'].dt.to_timestamp()
foods3_resample 

# Train-Validation-Test Split and Variable Transformation
#train-test split
n = len(foods1_resample)
train_df = foods3_resample[0:int(n*0.7)]
val_df = foods3_resample[int(n*0.7):int(n*0.9)]
test_df = foods3_resample[int(n*0.9):]

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
foods3_val_performance = {}
foods3_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
foods3_val_performance['Baseline - Last'] = baseline_last.evaluate(single_step_window.val)
foods3_performance ['Baseline - Last'] = baseline_last.evaluate(single_step_window.test, verbose=0)

# 1.1.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, single_step_window)
foods3_val_performance['Linear'] = linear.evaluate(single_step_window.val)
foods3_performance['Linear'] = linear.evaluate(single_step_window.test)

# 1.1.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, single_step_window)
foods3_val_performance['Dense'] = dense.evaluate(single_step_window.val)
foods3_performance['Dense'] = dense.evaluate(single_step_window.test)

# 1.1.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, single_step_window)
foods3_val_performance['LSTM'] = lstm_model.evaluate(single_step_window.val)
foods3_performance['LSTM'] = lstm_model.evaluate(single_step_window.test)
foods3_val_performance
foods3_val_performance

# 1.2. MULTI-STEP MODELS -------------------------------------------------------------------------------
# 1.2.1. Baseline Models

# Create dictionaries to store forecasts
ms_foods3_val_performance = {}
ms_foods3_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
ms_foods3_val_performance['Baseline - Last'] = baseline_last.evaluate(wide_window.val)
ms_foods3_performance ['Baseline - Last'] = baseline_last.evaluate(wide_window.test, verbose=0)

# 1.2.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, wide_window)
ms_foods3_val_performance['Linear'] = linear.evaluate(wide_window.val)
ms_foods3_performance['Linear'] = linear.evaluate(wide_window.test)

# 1.2.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, wide_window)
ms_foods3_val_performance['Dense'] = dense.evaluate(wide_window.val)
ms_foods3_performance['Dense'] = dense.evaluate(wide_window.test)

# 1.3.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, wide_window)
ms_foods3_val_performance['LSTM'] = lstm_model.evaluate(wide_window)
ms_foods3_performance['LSTM'] = lstm_model.evaluate(wide_window.test)
ms_foods3_val_performance
ms_foods3_val_performance


# II.2.  HOUSEHOLD ========================================================================================================================

# 1. HOUSEHOLD1 ------------------------------------------------------------------------------------------------
# Resample 
household1_resample = ProductDept_household1
household1_resample['yy-mm']  = household1_resample['Date'].dt.to_period('M')
household1_resample = household1_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
household1_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
household1_resample['Date'] = household1_resample['Date'].dt.to_timestamp()
household1_resample 

# Train-Validation-Test Split and Variable Transformation
#train-test split
n = len(household1_resample)
train_df = household1_resample[0:int(n*0.7)]
val_df = household1_resample[int(n*0.7):int(n*0.9)]
test_df = household1_resample[int(n*0.9):]

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
household1_val_performance = {}
household1_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
household1_val_performance['Baseline - Last'] = baseline_last.evaluate(single_step_window.val)
household1_performance ['Baseline - Last'] = baseline_last.evaluate(single_step_window.test, verbose=0)

# 1.1.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, single_step_window)
household1_val_performance['Linear'] = linear.evaluate(single_step_window.val)
household1_performance['Linear'] = linear.evaluate(single_step_window.test)

# 1.1.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, single_step_window)
household1_val_performance['Dense'] = dense.evaluate(single_step_window.val)
household1_performance['Dense'] = dense.evaluate(single_step_window.test)

# 1.1.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, single_step_window)
household1_val_performance['LSTM'] = lstm_model.evaluate(single_step_window.val)
household1_performance['LSTM'] = lstm_model.evaluate(single_step_window.test)
household1_val_performance
household1_val_performance

# 1.2. MULTI-STEP MODELS -------------------------------------------------------------------------------
# 1.2.1. Baseline Models

# Create dictionaries to store forecasts
ms_household1_val_performance = {}
ms_household1_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
ms_household1_val_performance['Baseline - Last'] = baseline_last.evaluate(wide_window.val)
ms_household1_performance ['Baseline - Last'] = baseline_last.evaluate(wide_window.test, verbose=0)

# 1.2.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, wide_window)
ms_household1_val_performance['Linear'] = linear.evaluate(wide_window.val)
ms_household1_performance['Linear'] = linear.evaluate(wide_window.test)

# 1.2.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, wide_window)
ms_household1_val_performance['Dense'] = dense.evaluate(wide_window.val)
ms_household1_performance['Dense'] = dense.evaluate(wide_window.test)

# 1.3.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, wide_window)
ms_household1_val_performance['LSTM'] = lstm_model.evaluate(wide_window)
ms_household1_performance['LSTM'] = lstm_model.evaluate(wide_window.test)
ms_household1_val_performance
ms_household1_val_performance

# 2. HOUSEHOLD2 ----------------------------------------------------------------------------------------------------------------------------
household2_resample = ProductDept_household2
household2_resample['yy-mm']  = household2_resample['Date'].dt.to_period('M')
household2__resample = household2_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
household2__resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
household2__resample['Date'] = foods2_resample['Date'].dt.to_timestamp()
household2__resample 

# Train-Validation-Test Split and Variable Transformation
#train-test split
n = len(household2__resample)
train_df = household2__resample[0:int(n*0.7)]
val_df = household2__resample[int(n*0.7):int(n*0.9)]
test_df = household2__resample[int(n*0.9):]

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
household2__val_performance = {}
household2__performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
foods2_val_performance['Baseline - Last'] = baseline_last.evaluate(single_step_window.val)
foods2_performance ['Baseline - Last'] = baseline_last.evaluate(single_step_window.test, verbose=0)

# 1.1.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, single_step_window)
foods2_val_performance['Linear'] = linear.evaluate(single_step_window.val)
foods2_performance['Linear'] = linear.evaluate(single_step_window.test)

# 1.1.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, single_step_window)
household2__val_performance['Dense'] = dense.evaluate(single_step_window.val)
household2__performance['Dense'] = dense.evaluate(single_step_window.test)

# 1.1.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, single_step_window)
household2__val_performance['LSTM'] = lstm_model.evaluate(single_step_window.val)
household2__performance['LSTM'] = lstm_model.evaluate(single_step_window.test)
household2__val_performance
household2__val_performance

# 1.2. MULTI-STEP MODELS -------------------------------------------------------------------------------
# 1.2.1. Baseline Models

# Create dictionaries to store forecasts
ms_household2__val_performance = {}
ms_household2__performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
ms_household2__val_performance['Baseline - Last'] = baseline_last.evaluate(wide_window.val)
ms_household2__performance ['Baseline - Last'] = baseline_last.evaluate(wide_window.test, verbose=0)

# 1.2.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, wide_window)
ms_household2__val_performance['Linear'] = linear.evaluate(wide_window.val)
ms_household2__performance['Linear'] = linear.evaluate(wide_window.test)

# 1.2.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, wide_window)
ms_household2__val_performance['Dense'] = dense.evaluate(wide_window.val)
ms_household2__performance['Dense'] = dense.evaluate(wide_window.test)

# 1.3.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, wide_window)
ms_household2__val_performance['LSTM'] = lstm_model.evaluate(wide_window)
ms_household2__performance['LSTM'] = lstm_model.evaluate(wide_window.test)
ms_household2__val_performance
ms_household2__val_performance


# II.3. HOBBIES ============================================================================================================================
# 1. HOBBIES1 ------------------------------------------------------------------------------------------------
# Resample 
hobbies1_resample = ProductDept_hobbies1
hobbies1_resample['yy-mm']  = hobbies1_resample['Date'].dt.to_period('M')
hobbies1_resample = hobbies1_resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
hobbies1_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
hobbies1_resample['Date'] = household1_resample['Date'].dt.to_timestamp()
hobbies1_resample 

# Train-Validation-Test Split and Variable Transformation
#train-test split
n = len(household1_resample)
train_df = hobbies1_resample[0:int(n*0.7)]
val_df = hobbies1_resample[int(n*0.7):int(n*0.9)]
test_df = hobbies1_resample[int(n*0.9):]

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
hobbies1_val_performance = {}
hobbies1_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
hobbies1_val_performance['Baseline - Last'] = baseline_last.evaluate(single_step_window.val)
hobbies1_performance ['Baseline - Last'] = baseline_last.evaluate(single_step_window.test, verbose=0)

# 1.1.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, single_step_window)
hobbies1_val_performance['Linear'] = linear.evaluate(single_step_window.val)
hobbies1_performance['Linear'] = linear.evaluate(single_step_window.test)

# 1.1.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, single_step_window)
hobbies1_val_performance['Dense'] = dense.evaluate(single_step_window.val)
hobbies1_performance['Dense'] = dense.evaluate(single_step_window.test)

# 1.1.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, single_step_window)
hobbies1_val_performance['LSTM'] = lstm_model.evaluate(single_step_window.val)
hobbies1_performance['LSTM'] = lstm_model.evaluate(single_step_window.test)
hobbies1_val_performance
hobbies1_val_performance

# 1.2. MULTI-STEP MODELS -------------------------------------------------------------------------------
# 1.2.1. Baseline Models

# Create dictionaries to store forecasts
ms_hobbies1_val_performance = {}
ms_hobbies1_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
ms_hobbies1_val_performance['Baseline - Last'] = baseline_last.evaluate(wide_window.val)
ms_hobbies1_performance ['Baseline - Last'] = baseline_last.evaluate(wide_window.test, verbose=0)

# 1.2.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, wide_window)
ms_household1_val_performance['Linear'] = linear.evaluate(wide_window.val)
ms_household1_performance['Linear'] = linear.evaluate(wide_window.test)

# 1.2.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, wide_window)
ms_hobbies1_val_performance['Dense'] = dense.evaluate(wide_window.val)
ms_hobbies1_performance['Dense'] = dense.evaluate(wide_window.test)

# 1.3.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, wide_window)
ms_hobbies1_val_performance['LSTM'] = lstm_model.evaluate(wide_window)
ms_hobbies1_performance['LSTM'] = lstm_model.evaluate(wide_window.test)
ms_hobbies1_val_performance
ms_hobbies1_val_performance

# 2. HOUSEHOLD2 ----------------------------------------------------------------------------------------------------------------------------
hobbies2_resample = ProductDept_hobbies2
hobbies2_resample['yy-mm']  = hobbies2_resample['Date'].dt.to_period('M')
hobbies2_resample = household2__resample.groupby(['yy-mm', 'Unemployment', 'Inflation', 'Real Disposable Income'])['Sales'].sum().reset_index()
hobbies2_resample.rename(columns={'yy-mm': 'Date'}, inplace=True)
hobbies2_resample['Date'] = foods2_resample['Date'].dt.to_timestamp()
hobbies2_resample 

# Train-Validation-Test Split and Variable Transformation
#train-test split
n = len(hobbies2_resample)
train_df = hobbies2_resample[0:int(n*0.7)]
val_df = hobbies2_resample[int(n*0.7):int(n*0.9)]
test_df = hobbies2_resample[int(n*0.9):]

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
hobbies1_val_performance = {}
household2_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
hobbies2_val_performance['Baseline - Last'] = baseline_last.evaluate(single_step_window.val)
hobbies2_performance ['Baseline - Last'] = baseline_last.evaluate(single_step_window.test, verbose=0)

# 1.1.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, single_step_window)
hobbies2_val_performance['Linear'] = linear.evaluate(single_step_window.val)
hobbies2_performance['Linear'] = linear.evaluate(single_step_window.test)

# 1.1.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, single_step_window)
hobbies2__val_performance['Dense'] = dense.evaluate(single_step_window.val)
hobbies2__performance['Dense'] = dense.evaluate(single_step_window.test)

# 1.1.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, single_step_window)
hobbies2_val_performance['LSTM'] = lstm_model.evaluate(single_step_window.val)
hobbies2_performance['LSTM'] = lstm_model.evaluate(single_step_window.test)
hobbies2_val_performance
hobbies2_val_performance

# 1.2. MULTI-STEP MODELS -------------------------------------------------------------------------------
# 1.2.1. Baseline Models

# Create dictionaries to store forecasts
ms_hobbies2_val_performance = {}
ms_hobbies2_performance = {}
# KEY: We keep adding to this at every type of model we instantiate and fit 

# Evaluate models on validation and test sets (i.e., forecast)
ms_hobbies2_val_performance['Baseline - Last'] = baseline_last.evaluate(wide_window.val)
ms_hobbies2_performance ['Baseline - Last'] = baseline_last.evaluate(wide_window.test, verbose=0)

# 1.2.2. Linear Model
linear = Sequential ([
    Dense(units=1)
])
history=compile_and_fit(linear, wide_window)
ms_hobbies2_val_performance['Linear'] = linear.evaluate(wide_window.val)
ms_hobbies2_performance['Linear'] = linear.evaluate(wide_window.test)

# 1.2.3. Dense Model
dense = Sequential([
    Dense(units=50, activation='relu'),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])
history = compile_and_fit(dense, wide_window)
ms_hobbies2_val_performance['Dense'] = dense.evaluate(wide_window.val)
ms_hobbies2_performance['Dense'] = dense.evaluate(wide_window.test)

# 1.3.4. LSTM Model
lstm_model = Sequential([
    Bidirectional(
    LSTM(128, return_sequences=True)),
    Dropout(rate=0.2),
    Dense(units=100), # set return_sequences to True to make sure past information is being used by the network
    Dense(units=1)
])
history = compile_and_fit(lstm_model, wide_window)
ms_hobbies2_val_performance['LSTM'] = lstm_model.evaluate(wide_window)
ms_hobbies2_performance['LSTM'] = lstm_model.evaluate(wide_window.test)
ms_hobbies2_val_performance
ms_hobbies2_val_performance


