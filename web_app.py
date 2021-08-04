from datetime import time
import streamlit as st
from src import time_series
from src import import_data
from src import data
from src.utils import *
from matplotlib import pyplot

# Get data

@st.cache
def get_ts_data(endog_file, endog_format, exog_file, exog_format, endog_name):
    '''Gets the PCA with percentage, or no PCA if None'''
    # endog_file = 'data/sample_tabular.xlsx'
    # exog_file = 'data/data.xlsx'
    # endog_name = 'EBITDA'
    # endog_format = import_data.EndogeneousDataFormats.excel_table
    # exog_format = import_data.ExogeneousDataFormats.macro

    endog = import_data.EndogeneousDataImporter.import_endogeneous(endog_file, endog_name, endog_format)

    exog = None if exog_file is None else import_data.ExogeneousDataImporter.import_exogeneous(exog_file, exog_format)
    
    return data.TimeSeriesData(exog, endog, endog_name)

@st.cache
def get_ts_data_after_downsample(ts_data, exogeneous_vars):
    '''Select exogeneous vars from ts_data'''
    return ts_data.downsample(exogeneous_vars)

@st.cache
def get_ts_data_after_PCA(ts_data, pca, pca_percentage):
    '''Takes a PCA of ts_data according to pca_percentage'''
    if pca:
        return ts_data.PCA(pca_percentage)
    else:
        return ts_data

st.sidebar.text('Choose files')
endog_filename = st.sidebar.file_uploader('Upload endogeneous data')
endog_file_type = st.sidebar.selectbox('Select endogeneous data file type', options=list(import_data.EndogeneousDataFormats))
exog_filename = st.sidebar.file_uploader('Upload exogeneous data')
exog_file_type = st.sidebar.selectbox('Select exogeneous data file type', options=list(import_data.ExogeneousDataFormats))
endog_var_name = st.sidebar.text_input('Endogeneous data name')


# ORIGINAL DATA
# for col selection purposes
ts_data_original = get_ts_data(endog_filename, endog_file_type, exog_filename, exog_file_type, endog_var_name)

# SIDEBAR
st.sidebar.text('Select columns')
exogeneous_vars = st.sidebar.multiselect('Select columns', ts_data_original.exogeneous_vars, default=ts_data_original.exogeneous_vars)
st.sidebar.text('Enable PCA')
pca = st.sidebar.checkbox('Enable PCA', value=True)
st.sidebar.text('PCA')
# TODO fix for vals 0 and 1
pca_percentage = st.sidebar.slider('select percentage explained variance by PCA', min_value=0.05, max_value=0.95, value=0.85, step=0.05)
st.sidebar.text('Model')
model = st.sidebar.selectbox('Select model', options=list(time_series.TimeSeriesRegressionModels))
st.sidebar.text('p d q order')
p = st.sidebar.number_input('p', min_value=0, value=0)
d = st.sidebar.number_input('d', min_value=0, value=0)
q = st.sidebar.number_input('q', min_value=0, value=0)
st.sidebar.text('P D Q order')
P = st.sidebar.number_input('P', min_value=0, value=0)
D = st.sidebar.number_input('D', min_value=0, value=0)
Q = st.sidebar.number_input('Q', min_value=0, value=0)
m = st.sidebar.number_input('m', min_value=0, value=4)
st.sidebar.text('Prediction horizon')
horizon = st.sidebar.number_input('horizon', min_value=0, max_value=len(ts_data_original.get_future_exogeneous()))
st.sidebar.text('Sliding window training size')
# TODO allow to 0 and 1
training_percentage = st.sidebar.slider('select percentage training data for sliding window forecast', min_value=0.05, max_value=1.0, value=0.80, step=0.05)

# GET DATA
ts_data_downsample = get_ts_data_after_downsample(ts_data_original, exogeneous_vars)
ts_data = get_ts_data_after_PCA(ts_data_downsample, pca, pca_percentage)
exogeneous_data = ts_data.get_exogeneous_data()
endogeneous_data = ts_data.endogeneous_data
future_exogeneous_data = ts_data.get_future_exogeneous()

if model == time_series.TimeSeriesRegressionModels.BruteForceARIMARegression:
    order_arg_ranges = (range(0, p+1),
        range(0, d+1),
        range(0, q+1),
        range(0, P+1),
        range(0, D+1),
        range(0, Q+1),
        range(m, m+1))
    ts = getattr(time_series, model.value)(endogeneous_data, exogeneous_data, order_arg_ranges)
else:
    ts = getattr(time_series, model.value)((p, d, q), (P, D, Q, m))

# PANEL
st.title('CMPC Financial Time Series Analysis')

# Endogeneous
st.header('Endogeneous data')
st.dataframe(endogeneous_data)
fig = ts_data.plot_endogeneous().figure
st.text('Time series plot of data')
st.pyplot(fig)
st.text('Autocorrelation plot of data')
st.pyplot(ts_data.plot_autocorrelation().figure)
st.text('ADF Test Differencing Order')
st.write(ts_data.ADF_test())
st.text('KPSS Test Differencing Order')
st.write(ts_data.KPSS_test())

# Exogeneous
st.header('Exogeneous_data')
st.dataframe(exogeneous_data)

# PCA
st.header('PCA Covariance matrix')
st.dataframe(ts_data_downsample.get_PCA_covariance_matrix())

# Fit
st.header('Model fit')
st.write(ts.fit(endogeneous_data, exogeneous_data))

# Predict
st.header('Model predict')
# TODO doesnt work with brute
st.write(ts.predict(horizon, endogeneous_data, exogeneous_data, future_exogeneous_data))

# Sliding window
st.header('Sliding window')
sliding_window = ts.sliding_window_forecast(endogeneous_data, exogeneous_data, training_percentage)
st.dataframe(sliding_window)
st.pyplot(sliding_window.plot().figure)

