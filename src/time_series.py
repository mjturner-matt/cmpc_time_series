import argparse
from ast import parse
from os import P_ALL, error, path
import sys

import math
from numpy.core.fromnumeric import repeat
from numpy.core.numeric import full

import pandas as pd
from pandas import plotting as pdplot
import numpy as np
from pandas.core.frame import DataFrame
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.metrics import mean_squared_error
from math import sqrt
from copy import deepcopy

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pmdarima.arima import auto_arima
from pmdarima import arima as pmd
from pmdarima import model_selection
from pmdarima.arima.utils import ndiffs

from matplotlib import pyplot as plt

from src.utils import *
from src.data import TimeSeriesData
from src.import_data import ExogeneousDataImporter, EndogeneousDataImporter, EndogeneousDataFormats, ExogeneousDataFormats
import enum


# display options
pd.set_option("display.max_columns", 999)
pd.set_option("display.max_rows", 999)

'''time_series.py
Tools for performing a time series regression.

This file contains various models that may be used to perform a time series regression, as listed
in TimeSeriesRegressionModels.  SARIMARegression is the base class and implements all public 
methods.  Its subclasses provide alternate models that may be used for regression.

Typical usage examples:
foo = SARIMARegression()
foo.fit(y, X)
bar = BruteForceARIMARegression(y, X, order_arg_ranges)
bar.predict(future_X)
'''

class TimeSeriesRegressionModels(enum.Enum):
    '''Represents models for time series regression.'''
    # SARIMARegression = SARIMARegression
    # AutoARIMARegression = AutoARIMARegression
    # BruteForceARIMARegression = BruteForceARIMARegression
    SARIMARegression = 'SARIMARegression'
    AutoARIMARegression = 'AutoARIMARegression'
    BruteForceARIMARegression = 'BruteForceARIMARegression'

class OverspecifiedError(Exception):
    '''
    Error for models which are overspecified, that is, the number of regressors is greater than or equal to 
    the number of observations.
    '''


class SARIMARegression():
    '''
    Represents an immutable SARIMA time series regression model.

    The order of the model is fixed at initialization and cannot be altered.
    
    The model provides various methods to invoke on endogeneous and exogeneous data, 
    but the underlying model parameters remain the same.  

    Attributes:
    p_d_q_order: the order of the model in the form (p,d,q)
    P_D_Q_order: the seasonal order of the model in the form (P,D,Q,m)
    '''
    # Abstraction function:
    # AF(p_d_q_order, P_D_Q_order) = A SARIMA regression model of order p_d_q_order and seasonal
    #     order P_D_Q_order
    # Rep invariant:
    # - p_d_q_order is a length three iterable of whole numbers (Z+)
    # - P_D_Q_order is a length four iterable of whole numbers (Z+)
    # - P_D_Q_order[3] is a natural number >=2
    # - total_order= sum(p_d_q_order) + sum(P_D_Q_order[0:3])
    # Safety from rep exposure:
    # - p_d_q order and P_D_Q_order are immutable types
    # - No mutator methods
    # - Getter methods for attributes use the @property decorator so no reassignments can be made
    
    max_iter = 100

    def __init__(self, p_d_q_order : tuple=(0,0,0), P_D_Q_order : tuple=(0,0,0,4)):
        '''
        Instantiates a new SARIMARegression object of the given order.

        Keyword arguments:
        p_d_q_order -- the order of the regression of the form (p, d, q).
            Requires p,d, and q to be >=0.
        P_D_Q_order -- the order of the regression of the form (P, D, Q, m).
            Requires P,D, and Q to be >=0.  Requires m>=2
        
        Raises:
        ValueError if the model contains any seasonal and ordinary lag of the same order.
        '''
        # use construction to test for valueerror
        # model = pmd.ARIMA(order=p_d_q_order, 
        #                     seasonal_order=P_D_Q_order,
        #                     maxiter=self.max_iter,
        #                     trend='ct',
        #                     enforce_stationarity=True,
        #                     enforce_invertibility=True,
        #                     measurement_error=False)
        # model = SARIMAX(np.ones(100), order=p_d_q_order, seasonal_order=P_D_Q_order, trend='ct')
        repeat_lags = self._find_repeat_lags(p_d_q_order, P_D_Q_order)
        if len(repeat_lags) >0:
            raise ValueError('Repeat ordinary and seasonal lags of orders %s.' %repeat_lags)

        self._p_d_q_order = p_d_q_order
        self._P_D_Q_order = P_D_Q_order
        self._total_order = p_d_q_order[0] + p_d_q_order[2] + P_D_Q_order[0] + P_D_Q_order[2]

        # TODO refactor to instantiate model here
        
        self._checkrep()

    def _checkrep(self):
        '''Asserts the rep invariant'''
        # Length 3 tuple
        assert len(self._p_d_q_order) == 3
        assert len(self._P_D_Q_order) == 4

        # whole numbers
        for i in self._p_d_q_order:
            assert i >= 0
        for j in self._P_D_Q_order:
            assert j >= 0
        
        # natural number
        assert self._P_D_Q_order[3] >=2

        # sum
        assert self._total_order == self._p_d_q_order[0] + self._p_d_q_order[2] + self._P_D_Q_order[0] + self._P_D_Q_order[2]

    @property
    def p_d_q_order(self):
        '''Order of the model of the form (p,d,q)'''
        return tuple(self._p_d_q_order)

    @property
    def P_D_Q_order(self):
        '''Order of the model of the form (P,D,Q,m)'''
        return tuple(self._P_D_Q_order)


    def _fit_model(self, endogeneous_data, exogeneous_data, p_d_q_order, P_D_Q_order):
        '''
        Fits the SARIMAX model to the data.

        Keyword arguments:
        endogeneous_data -- array-like, the regressor
        exogeneous_data -- array-like, the regressand (Optional).  Length must be equal to the length of endogeneous_data.

        Returns:
        An ARIMA model fit object
        '''
        ar = SARIMAX(endog=endogeneous_data, exog=exogeneous_data, 
                        order=p_d_q_order,
                        seasonal_order=P_D_Q_order,
                        trend='ct',
                        measurement_error=False,
                        enforce_stationarity=True,
                        enforce_invertibility=True)
        model_fit = ar.fit(cov_type='robust', maxiter=self.max_iter, full_output=False, disp=False)
        return model_fit

    def _sliding_window(self, endogeneous_data, exogeneous_data, p_d_q_order, P_D_Q_order, train_size : int):
        '''
        Conducts a sliding window forecast using the model.  
        
        For each possible block of consecutive time periods of size train_size,
        uses the first n-1 of the n-sized block as training data and tests using the last observation.

        Keyword arguments:
        y -- the endogeneous data Series in float64 format with a PeriodIndex.  Requires len(y) >=1.
        X - the exogeneous DataFrame in float64 format with a PeriodIndex (Optional).  If None, regression 
            only including endogeneous data and lags is conducted.
            Requires len(y) == len(X).
        p_d_q_order -- the order of the model in (p,d,q) format.
        P_D_Q_order -- the seasonal order of the model in (P,D,Q,m) format.
        train_size -- the length of the training window.  Requires 1 <= train_size < len(y)
            and train_size to be large enough that the model is not overspecified.
            # TODO define these terms better

        Returns:
        A tuple of the form (yhats, actuals) of numpy arrays, each of length len(y) - train_size
        containing no infs or nulls.  yhats[i] represents the prediction for the (len(y) - 1 - train_size + i)th 
        element of y, and actual[i] = y[len(y) - 1 - train_size + i].

        Raises:
        # TODO does it raise valueError if lags of same order???
        '''
        n = len(endogeneous_data)

        yhats = list()
        actuals = list()

        for train_start_index in range(n- train_size):
            # Get data
            endog_data = endogeneous_data.iloc[train_start_index : train_start_index + train_size]

            # get next values for prediction
            next_endog = endogeneous_data.iloc[train_start_index + train_size]

            if exogeneous_data is not None:
                exog_data = exogeneous_data.iloc[train_start_index: train_start_index + train_size, :]
                next_exog = pd.DataFrame(exogeneous_data.iloc[train_start_index + train_size]).transpose()
            else:
                exog_data = None
                next_exog = None
            # run model
            fit = self._fit_model(endog_data, exog_data, p_d_q_order, P_D_Q_order)
            # returns as series, convert to float
            yhat = fit.forecast(n_periods=1, exog=next_exog).iloc[0]

            yhats.append(yhat)
            actuals.append(next_endog)

        return (yhats, actuals)

    def _check_overspecified(self, y, X):
        '''
        Checks whether the model is overspecified.

        The model is overspecified if the total number of autoregressive and moving average lags (both normal and seasonal)
        plus the number of X variables plus the added variable for the time trend 
        is greater than or equal to the number of observations in the model.

        Keyword arguments:
        y -- the endogeneous data.
        X -- the exogeneous data.  Requires len(X) == len(y)

        Returns: 
        True if the model would be overspecified when fit for y on X, false otherwise.
        '''
        num_exog_vars = 0 if X is None else X.shape[1]
        # TODO refactor as constant for class
        trend_order =1
        # x vars + AR + MA + trend
        if not num_exog_vars + self._total_order +trend_order < len(y):
            return True

        return False

    def _find_repeat_lags(self, p_d_q_order, P_D_Q_order):
        '''
        Finds repeat AR or MA lags in the ordinary or seasonal components.

        Keyword arguments:
        p_d_q_order -- the order of the model in (p,d,q) format.
        P_D_Q_order -- the seasonal order of the model in (P,D,Q,m) format.

        Returns: 
        The set of all repeat lags found in the model.
        '''
        # AR lags
        ar_lags = set([*range(1,p_d_q_order[0]+1)])
        seasonal_ar_lags = set(np.array([*range(1, P_D_Q_order[0]+1)])
                               * P_D_Q_order[3])
        duplicate_ar_lags = ar_lags.intersection(seasonal_ar_lags)
        # MA lags
        ma_lags = set([*range(1,p_d_q_order[2]+1)])
        seasonal_ma_lags = set(np.array([*range(1, P_D_Q_order[2]+1)])
                               * P_D_Q_order[3])
        duplicate_ma_lags = ma_lags.intersection(seasonal_ma_lags)

        return duplicate_ar_lags.union(duplicate_ma_lags)

    def fit(self, y : pd.Series, X) -> str:
        '''
        Fits the model.
        
        Keyword arguments:
        y -- the endogeneous data Series in float64 format with a PeriodIndex.  Requires len(y) >= 0.
        X - the exogeneous DataFrame in float64 format with a PeriodIndex.  Requires len(y) == len(X)
            If no exogeneous data, pass an empty DataFrame containing the required index.

        Returns:
        A string summary of the results.

        Raises:
        OverspecifiedError if the model is overspecified, that is, if the number of provided exogeneous variables
            plus the number of lags (seasonal and otherwise) and trend order are greater than or equal to the number
            of exogeneous variables.
        '''
        assert len(y) == len(X), 'y and X must be of the same length'
        if self._check_overspecified(y, X):
            raise OverspecifiedError('Model is overspecified.  Remove exogeneous vars or reduce order')
        # already covered by check overspecified
        # assert len(y) >=1, 'Cannot fit a model to empty y (endogeneous data)'

        # TODO temporary patch, this should be spec of underlying fit method
        if len(X.columns) == 0:
            X = None
        else:
            X = X
        self._checkrep()
        fit = self._fit_model(y, X, self._p_d_q_order, self._P_D_Q_order)
        return fit.summary()

    def _predict(self, y, X, future_X, n):
        '''Helper method for predict'''
        fit = self._fit_model(y, X, self._p_d_q_order, self._P_D_Q_order)


        return fit.forecast(n, exog=future_X)


    def predict(self, n, y, X, future_X=None) -> pd.Series:
        '''
        Predicts n future values of y using future values of X, future_X.  
        Uses a model fit for y on X and then predicts using that model.

        Keyword arguments:
        n -- int, the number of future periods to predict.  Requires n>=0.
        y -- the endogeneous data Series in float64 format with a PeriodIndex.
        X - the exogeneous DataFrame in float64 format with a PeriodIndex.  Requires len(y) == len(X)
            If no exogeneous data, pass an empty DataFrame containing the required index.
        future_X -- Dataframe containing the future values of the exogeneous variables X in float64 format. (Optional)
            Requires the same set of columns in the same order as X, and with a PeriodIndex matching in 
            frequency and containing subsequent consecutive periods to the last periods in X, with no gaps.
            Required if X is provided, and must be of length >=n.

        Returns:
        A Pandas Series indexed by PeriodIndex in float64 format containing n future predictions for y.

        Raises:
        OverspecifiedError if the model is overspecified, that is, if the number of provided exogeneous variables
            plus the number of lags (seasonal and otherwise) and trend are greater than or equal to the number
            of exogeneous variables.
        ValueError if n is larger than the number of X observations provided.
        '''
        if self._check_overspecified(y, X):
            raise OverspecifiedError('Model is overspecified.  Remove exogeneous vars or reduce order')
        # TODO reinstate
        # assert not (X is None ^ future_X is None), 'X and future_X must either be both provided or both None' 

        # TODO temporary patch, this should be spec of underlying fit method
        if len(X.columns) == 0:
            X = None
        else:
            X = X
        start_date = y.index.max() + y.index.freq
        index = pd.period_range(start=start_date, periods=n, freq=y.index.freq)
        future_X = None if future_X is None else future_X.iloc[:n]
        if n >0:
            return pd.Series(data=self._predict(y,X,future_X,n),
                                index=index,
                                name='Predictions',
                                dtype='float64')
        else:
            return pd.Series(
                                index=pd.PeriodIndex(data=np.array([]), freq=y.index.freq),
                                name='Predictions',
                                dtype='float64'
            )

    def sliding_window_forecast(self, y : pd.Series, X, window_size=0.8) -> pd.DataFrame:
        '''
        Conducts a sliding window forecast using the model.  
        
        For each possible block of consecutive time periods of size approximately window_size * len(y), 
        uses the first n-1 of the n-sized block as training data and tests using the last observation.

        Keyword arguments:
        y -- the endogeneous data Series in float64 format with a PeriodIndex.
        X - the exogeneous DataFrame in float64 format with a PeriodIndex (Optional).  If None, regression only including endogeneous 
            data and lags is conducted.
            Requires len(y) == len(X).
        window_size -- the fraction of the total data to use for the training window.
                        Uses ceil(window_size*len(y)) of the data to train.
                        Requires 0 < window_size <= 1
                        Default 0.8.

        Returns:
        A Pandas DataFrame indexed by period of size approximately (1-window_size) * len(y) of two columns, 'predictions',
        which contains the predicted values by rolling window forecast, and 'actuals', which contains
        the actual y values for those periods.

        Raises:
        OverspecifiedError if window_size results in a window size that is overspecified in the current model.
        '''
        # assert precondition - fail fast
        assert 0 < window_size <= 1
        # TODO update underlying method
        window_len = max(math.ceil(window_size * len(y)), 1)
        if len(X.columns) == 0:
            X = None
            X_check = None
        else:
            X = X
            X_check = X.iloc[:window_len]
        if self._check_overspecified(y.iloc[:window_len], X_check):
            raise OverspecifiedError('window_size results in a model that is overspecified.')

        n_predictions = len(y) - window_len

        yhats, actuals = self._sliding_window(y, X, self._p_d_q_order, self._P_D_Q_order, window_len)

        self._checkrep()
        idx = y.index[len(y)-n_predictions:]
        return pd.DataFrame({'predictions': yhats, 'actuals': actuals}, index=idx, dtype='float64')

    
class AutoARIMARegression(SARIMARegression):
    '''
    Represents an immutable Auto ARIMA time series regression model.

    The order of the model is selected automatically at each fit, prediction, or sliding_window_forecast.
    However, each fit begins at the same (either default or provided) parameters for p, q, P, Q, and m.
    If provided, d and D are taken as given and not altered for any fit of the model.
    
    The model provides various methods to invoke on endogeneous and exogeneous data, 
    but the underlying model automatic ARIMA regression model remains the same.

    Attributes:
    p_d_q_order: the order of the model in the form (p,d,q)
    P_D_Q_order: the seasonal order of the model in the form (P,D,Q,m)
    '''
    # Abstraction function:
    # AF(p_d_q_order, P_D_Q_order) = an AutoARIMA time series regression using starting parameters
    #     p=p_d_q_order[0], q=p_d_q_order[2], P=P_D_Q_order[0], Q=P_D_Q_order[2] and 
    #     fixed differencing parameters d=p_d_q_order[1] and D=P_D_Q_order[1], or autoamitaclly set
    #     parameters if p_d_q_order or P_D_Q_order are None.  m=P_D_Q_order[3] is the seasonal order.
    # Rep invariant:
    # - p_d_q is a length 3 iterable of whole numbers (Z+).  Only p_d_q[1] can be None.
    # - P_D_Q_order is a length 4 iterable of whole numbers (Z+).  Only P_D_Q[1] can be None
    # - P_D_Q_order[3] is a whole number >=2
    # - total_order = 0
    # Safety from rep exposure:
    # - p_d_q_order and P_D_Q_order are immutable types
    # - No mutator methods
    # - Getter methods for attributes use the @property decorator so no reassignments can be made
    
    def __init__(self, p_d_q_order : tuple=None, P_D_Q_order : tuple=None):
        '''
        Initiates an AutoARIMARegression object.

        Keyword arguments:
        p_d_q_order -- The starting order for the ARIMA model.  Note that p and q are provided
            only as starting parameters and may be optimized by the model.  d will be used
            as given and will not be altered by the model.  If None, p, d, and q will be determined
            automatically.
        P_D_Q_order -- The starting order of the seasonal component of the SARIMA model.  Note that P
            and Q are starting parameters only and may be optimized by the model.  D will be used
            as given and will not be altered by the model.  If None, P, D, and Q will be determined 
            automatically.
        '''
        if p_d_q_order:
            self._p_d_q_order = p_d_q_order
        else:
            # Based on default vals for params, see pdarima docs
            self._p_d_q_order = (2, None, 2)
        
        if P_D_Q_order:
            self._P_D_Q_order = P_D_Q_order
        else:
            # Based on default vals for params, see pdarima docs
            self._P_D_Q_order = (1, None, 1, 4)

        self._total_order = 0

        self._checkrep()

    def _checkrep(self):
        '''Asserts the rep invariant'''
        # p, q, P, Q
        z_plus_indices = [0,2]
        for i in z_plus_indices:
            assert self._p_d_q_order[i] >= 0
            assert self._P_D_Q_order[i] >= 0

        # d, D - Can be None to let model auto set differencing
        if self._p_d_q_order[1] != None:
            assert self._p_d_q_order[1] >=0
        if self._P_D_Q_order[1] != None:
            assert self._P_D_Q_order[1] >=0
        
        # Seasonal order >=2
        assert self._P_D_Q_order[3] >=2

        # total order = 0
        assert self._total_order == 0

    def _fit_model(self, endogeneous_data, exogeneous_data, p_d_q_order, P_D_Q_order):
        '''
        Fits the auto ARIMA model to the data.

        Keyword arguments:
        endogeneous_data -- array-like, the regressor
        exogeneous_data -- array-like, the regressand.  Length must be equal to the length of endogeneous_data

        Returns:
        An ARIMA model fit object
        '''
        p, d, q = p_d_q_order
        P, D, Q, m = P_D_Q_order

        model_fit = auto_arima(endogeneous_data, exogeneous_data, start_p=p, d=d, start_q=q, start_P=P, D=D, start_Q=Q, m=m, trend='t', maxiter=self.max_iter, sarimax_kwargs={'enforce_stationarity':True, 'enforce_invertibility':True}, **{'cov_type':'robust', 'gls':False})

        return model_fit

    def _predict(self, y, X, future_X, n):
        fit = self._fit_model(y, X, self.p_d_q_order, self.P_D_Q_order)

        return fit.predict(n_periods=n, X=future_X)

    def _sliding_window(self, endogeneous_data, exogeneous_data, p_d_q_order, P_D_Q_order, window_size):
        '''Rolling window auto arima forecast'''
        if window_size == len(endogeneous_data):
            return (np.array([]), np.array([]))
        else:
            y = endogeneous_data
            x = exogeneous_data
            cv = model_selection.SlidingWindowForecastCV(h=1, window_size=window_size)
            fit = self._fit_model(y, x, p_d_q_order, P_D_Q_order)
            yhats = model_selection.cross_val_predict(fit, y, x, cv=cv)
            actuals = y.iloc[-len(yhats):].to_numpy()

            return (yhats, actuals)

class BruteForceARIMARegression(SARIMARegression):
    '''
    Represents an immutable brute force regression model.

    The order of the ARIMA model is set according to a brute force search over all 
    combinations of orders provided in order_args.  After the optimal is chosen upon
    initialization, it is immutable and cannot be changed.
    
    The model provides various methods to invoke on endogeneous and exogeneous data, 
    but the underlying model parameters remain the same.  

    Attributes:
    p_d_q_order: the order of the model in the form (p,d,q)
    P_D_Q_order: the seasonal order of the model in the form (P,D,Q,m)
    '''
    # Abstraction function:
    # AF(y, X, order_arg_ranges): A SARIMAX time series regression based on the 
    #     order arguments chosen using brute force search over all parameters in 
    #     order_arg_ranges that result in the lowest rmse rolling window forecast
    #     predictive error on a regression of y on X.
    # Rep invariant:
    # - p_d_q_order is a length three iterable of whole numbers (Z+)
    # - P_D_Q_order is a length four iterable of whole numbers (Z+)
    # - P_D_Q_order[3] is a natural number 
    # Safety from rep exposure:
    # - p_d_q order and P_D_Q_order are immutable types
    # - No mutator methods
    # - Getter methods for attributes use the @property decorator so no reassignments can be made
    
    fraction_training_data = 0.8

    def __init__(self, y, X, order_arg_ranges):
        '''
        Initializes a new class instance.

        Keyword arguments:
        y -- the endogeneous data.  Requires len(y) >=2
        X -- the exogeneous data.  Requires len(y) == len(X)
        order_arg_ranges -- a tuple of range objects of the form (p, d, q, P, D, Q, m)
            where p, d, q, P, D, Q, and m are the ranges for their respective order arguments.

            Example:
                ((range(0,4), #p can be 0,1,2,3
                range(0,3), #d
                range(0,4), #q 
                range(0,2), #P
                range(0,2), #D
                range(0,2), #Q
                range(4,5)) #m

        Raises:
        OverspecifiedError if all models allowed by order_arg_ranges would be overspecified.
        ValueError if all models allowed contain seasonal and ordinary lag of the same order.
        '''
        # Determine the size of the training window when performing rolling
        # forecasting for the optimal
        # assert precondition on X
        assert len(y) == len(X)

        n = len(y)
        # TODO maybe call rolling_window which calculates window size automatically
        training_window_size = min(max(math.ceil(self.fraction_training_data * n), 1), len(y)-1)
        order = self.determine_opt_order(y, X, training_window_size, order_arg_ranges)
        self._p_d_q_order = tuple(order[0:3])
        self._P_D_Q_order = tuple(order[3:])

        self._total_order = self._p_d_q_order[0] + self._p_d_q_order[2] + self.P_D_Q_order[0] + self._P_D_Q_order[2]
        self._checkrep()

    def determine_opt_order(self, y, X, training_window_size, order_arg_ranges):
        '''
        Determines the optimal order for the SARIMA model y on X.

        Uses a brute force search over all combinations of parameters in
        order_arg_ranges to find which rolling window forecast of y on X of 
        size training_window_size results in the lowest rmse error on the predictions.

        Keyword arguments:
        y -- the endogeneous data Series in float64 format with a PeriodIndex.  Requires len(y) >= 2.
        X - the exogeneous DataFrame in float64 format with a PeriodIndex.  Requires len(y) == len(X)
            If no exogeneous data, pass an empty DataFrame containing the required index.
        training_window_size -- the size of the training window.  Requires training_window_size < len(y)
        order_arg_ranges -- a tuple of range objects of the form (p, d, q, P, D, Q, m)
            where p, d, q, P, D, Q, and m are the ranges for their respective order arguments.

            Example:
                ((range(0,4), #p can be 0,1,2,3
                range(0,3), #d
                range(0,4), #q 
                range(0,2), #P
                range(0,2), #D
                range(0,2), #Q
                range(4,5)) #m

        Returns:
        A tuple of the form (p,d,q,P,D,Q,m) where each element is the optimal 
        whole number order argument.

        Raises:
        OverspecifiedError if all models allowed by order_arg_ranges would be overspecified.
        ValueError if all models allowed contain seasonal and ordinary lag of the same order.
        '''
        trend_order = 1
        min_eval_obs = 1
        # Fail fast - test if overspecified here.
        # TODO take min of ranges
        min_order_args_order = order_arg_ranges[0][0] + order_arg_ranges[2][0] + order_arg_ranges[3][0] + order_arg_ranges[5][0]
        if not min_order_args_order + trend_order + X.shape[1] + min_eval_obs < len(y):
            raise OverspecifiedError('Model is overspecified for every combination of order args given by order_arg_ranges.  Remove exogeneous vars or reduce order')
        min_rmse = float('inf')
        best_order_arg = None
        if X.shape[1] == 0:
            X = None
        for order_arg in range_combo_generator(order_arg_ranges):
            p_d_q_order = tuple(order_arg[0:3]) # p,d,q
            P_D_Q_order = tuple(order_arg[3:]) # P, D, Q, m
            # if it's a valid model, try
            if not len(self._find_repeat_lags(p_d_q_order, P_D_Q_order)) >0:
                try:
                    yhats, actuals = self._sliding_window(y, X, p_d_q_order, P_D_Q_order, training_window_size)
                    rmse = calc_rmse(yhats, actuals)
                except OverspecifiedError:
                    rmse = float('inf')
                if rmse < min_rmse:
                    min_rmse = rmse
                    best_order_arg = order_arg

        # if best_order_arg is None:
        #     raise ValueError('All models allowed by order_args contain seasonal and ordinary lags of the same order.')
        return best_order_arg

        


if __name__ == '__main__':
    None


    ############################
    # Example with quarterly data
    # Get data

    # cmpc_endog = data.get_endogeneous_data()
    # cmpc_exog = data.PCA().get_exogeneous_data()

    # window_size = 32

    # ARIMA model
    # cmpc_arima = SARIMARegression((2,1,3), (0,0,1,4))
    # cmpc_arima = SARIMARegression
    #((0,0,0), (0,0,0,4))
    # print(cmpc_arima.fit(cmpc_endog, cmpc_exog))
    # print(cmpc_arima.rolling_window_forecast(cmpc_endog, cmpc_exog, window_size))
    # print(cmpc_arima.p_d_q_order)

    # AutoARIMA model
    # cmpc_auto_arima = AutoARIMARegression()
    # print(cmpc_auto_arima.fit(data.get_endogeneous_data(), data.PCA().get_exogeneous_data()))
    # print(cmpc_auto_arima.rolling_window_forecast(cmpc_endog, cmpc_exog, window_size))
    # print(cmpc_auto_arima.n_diffs(cmpc_endog))

    # BruteForceARIMA model
    # order_arg_ranges = (range(0,1),
    #                 range(0,1),
    #                 range(0,1),
    #                 range(0,1),
    #                 range(0,1),
    #                 range(0,1))
    # order_arg_ranges =  (range(0,4), #p - Cannot equal m
    #         range(0,3), #d
    #         range(0,4), #q - Cannot equal m
    #         range(0,2), #P
    #         range(0,2), #D
    #         range(0,2)) #Q
    # cmpc_brute_arima = BruteForceARIMARegression(cmpc_endog, cmpc_exog, order_arg_ranges)
    # print(cmpc_brute_arima.fit(cmpc_endog, cmpc_exog))
    # to_excel(cmpc_brute_arima.rolling_window_forecast(cmpc_endog, cmpc_exog, 32), 'brute_results.xlsx')


    # new argparse design
    # automaticlaly parse file and put in dataframe
    # --macro_file --financial_file --endog_var
    # --pca to allow for principal components 
    # --create components db , maybe same arg as above

    # --method: sarima, auto arima, brute
    # --method order 
    # fit, rolling window, predict
    # could print, save file, or plot

    # design 2 
    # import file with options, 
    # time series with options
    # file that puts it all together