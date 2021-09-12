# support for type hints for self/producer methods like PCA
from __future__ import annotations
# import argparse
# from ast import parse
# from os import P_ALL, error, path
# import sys

# import math
# from numpy.core.numeric import full

import pandas as pd
from pandas import plotting as pdplot
import numpy as np
from pmdarima import arima
from copy import deepcopy

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# from pmdarima.arima import auto_arima
# from pmdarima import model_selection
# from pmdarima.arima.utils import ndiffs

from matplotlib import pyplot as plt

# display options
pd.set_option("display.max_columns", 999)
pd.set_option("display.max_rows", 999)

'''Tools to represent a time series dataset.

TimeSeriesData represents a time series dataset, including both endogeneous and exogeneous and 
past and future data.  Allows for the transformation of data features to create 
a transformed time series dataset of the same datatype.

Typical usage examples:
foo = TimeSeriesData(exogeneous_data, endogeneous_data, endogeneous_var)
foo.exogeneous_data()
foo.PCA()
'''

class IntersectionError(Exception):
    """Exception raised for errors in the intersection of exogeneous and endogeneous data.

    Attributes:
        message -- explanation of the error
    """

class TimeSeriesData():
    '''
    Represents an immutable time series dataset.

    Represents endogenous data and its corresponding exogeneous data, and 
    future predicted exogeneous data to allow for forecasting.  Enforces 
    that there exists at least one endogeneous time series observation, and that no 
    variable is either infinite or null.

    Enforces the following properties on the time series dataset:
    - There exists an observation for the endogeneous variable in at least one time period
    - There are no "gaps" or missing observations of the endogeneous variable between the first
        and last ones provided
    - Each exogeneous variable is observed (is not missing) for every time period that 
        the endogeneous variable is observed
    - A future period is defined as a period which occurs after the last observation of the 
        endogeneous variable.  For every future period in which any exogeneous variable has 
        an observation, all exogeneous variables have an observation
        for that period.  Define future period 0 as the period occurring directly after the last
        observation of the endogeneous variable.  Having an observation for future period i+1 implies having 
        an observation for future period i.

    Attributes:
    _exogeneous_vars: Names of all exogeneous vars
    endogeneous_data: Endogeneous data
    '''
    # Abstraction function:
    # AF(exogeneous_data, endogeneous_data, endogeneous_var): The periodic time series of the 
    #   intersection of exogeneous_data and endogeneous_data, followed by any subsequent
    #   observations of exogeneous_data as future_data
    # Rep invariant:
    # 1. Indices of both db and future db are both period indices of consecutive periods with no missing or duplicate values
    # 2. If nonempty, the index of future_db begins exactly one period after the index of db
    # 3. No NaN or infinite values in either db or futue_db.  No NaN values in either index.
    # 4. Endogeneous var is in columns of db, but not in future_db
    # 5. Exogenous_var is list of unique elements where x is in exogeneous vars iff x is 
    #     in self.db._columns and x != endogeneous_var
    # 6. x in _exogeneous_vars iff x in columns(future_db)
    # 7. db is of length >=1
    # 8. db and future_db are of the same freqstr
    # TODO dtype in float64
    # Safety from rep exposure:
    # - Immutable class with no mutator methods
    # - All producers and observers make defensive copies
    # - PCA makes new instance without defensive copy, but this is ok
    #     since underlying data is still protected by the safety from rep exposure of this class

    # Constructors
    def __init__(self, exogeneous_data : pd.DataFrame, endogeneous_data : pd.Series, endogeneous_var : str):
        '''
        Instantiates a TimeSeriesData object.

        Takes the intersection of exogeneous_data and endogeneous_data, but maintains future exogeneous
        data, if any.  Define the starting period as the first period in the intersection between
        exogeneous_data and endogeneous_data.  Define the current period as the last period in the intersection
        between exogeneous_data and endogeneous_data, and the last period as the last period containing exogeneous
        data observations.  Any exogeneous data ocurring after the current period up to the last period is
        future data.

        Keyword arguments: 
        exogeneous_data -- Dataframe indexed by datetime where each column contains 
            a unique exogeneous variabe in float64 format.  The dataframe contains no infinite or null values.
            Data is sorted in ascedning order by date.  Must be of length at least 1.
            If no exogeneous data, use None instead.
        endogeneous_data -- A series of endogeneous_var of type float64, indexed by period and sorted
            in increasing date order, where no entry is infinite or null, no period is duplicated, 
            and no intermediate period is missing.  Must contain at least one observation.
        endogeneous var -- the name of the endogeneous variable.  Requires endogeneous_var to be name of 
            endogeneous_data, and endogeneous_var may not be in exogeneous_data.

        Raises:
        ValueError if frequency of exogeneous_data is less than the frequency of endogeneous_data, 
            that is, if for any time period in the frequency of endogeneous data between
            the starting period and the last period in which exogeneous_data does not 
            contain an observation.
        IntersectionError if the intersection of the two datasets is the empty set.
        '''
        # assert preconditions on length
        assert len(endogeneous_data) >= 1, 'Must have at least one observation of endogeneous data'
        # assert naming convention
        assert endogeneous_data.name == endogeneous_var
        if exogeneous_data is not None:
            assert endogeneous_var not in exogeneous_data.columns, 'Endogeneous var name found in exogeneous data'

        self._db, self._future_db = self._import_data(exogeneous_data, endogeneous_data, endogeneous_var)
        self.endogeneous_var = endogeneous_var
        cols = list(self._db.columns)
        cols.remove(self.endogeneous_var)
        self._exogeneous_vars = deepcopy(cols)

        self._checkrep()

    @classmethod
    def _from_db(cls, db : pd.DataFrame, future_db : pd.DataFrame, endogeneous_var : str) -> TimeSeriesData:
        '''
        Internal method to instantiate a new instance of the class from an 
        existing internal representation.

        Keyword arguments:
        db -- the internal database, which must comply with the rep invariant.
        future_db -- the internal database of future exogeneous data, which 
            must comply with the rep invariant
        endogeneous_var -- the column name of the endogeneous variable, 
            which must comply with the rep invariant.
        '''
        ob = cls.__new__(cls)
        ob.endogeneous_var = endogeneous_var
        ob._db = db
        ob._future_db = future_db
        cols = list(db.columns)
        cols.remove(endogeneous_var)
        ob._exogeneous_vars = cols
        ob._checkrep()
        return ob

    def _checkrep(self):
        '''Asserts the rep invariant'''
        dbs = [self._db, self._future_db]
        for db in dbs:
            if len(db) > 0:
                # 1. No missing quarters
                # Sufficient to show:
                # No duplicate quarterly observations
                assert not np.any(db.index.duplicated())
                # Every quarterly observation between max and min filled:
                time_delta = db.index.max() - db.index.min()
                # should be a +1 on time delta since we have 40 periods, but observation at beginning of first period
                # example: quarter 1-quarter 0 =1, although we have 2 quarters
                assert len(db.index) == time_delta.n +1

        # 2. Index of future db occurs exactly one period after db
        if len(self._future_db) >0:
            db_index_delta = self._future_db.index.min() - self._db.index.max() 
            assert db_index_delta.n == 1

        # 3. No NaN or infinite values values.  No NaN's in either index.
        for db in dbs:
            assert not np.any(db.isnull())
            assert not np.any(np.isinf(db))
        for index in [getattr(db, 'index') for db in dbs]:
            assert not np.any(index.isnull())

        # 4. endogeneous var in columns of db, but not future_db
        assert self.endogeneous_var in self._db.columns
        assert self.endogeneous_var not in self._future_db.columns

        # 5. _exogeneous_vars is unique list where
        # x in exogenous var iff x is in self._db.columns and x is not endogeneous_var
        # no duplicates already asserted by pandas
        cols_set = set(self._db.columns)
        cols_set.remove(self.endogeneous_var)
        exog_set = set(self._exogeneous_vars)
        assert cols_set == exog_set

        # 6. x in _exogeneous_vars iff x in cols(future_db)
        for i in self._exogeneous_vars:
            assert i in self._future_db.columns
        assert len(self._exogeneous_vars) == len(self._future_db.columns)

        # 7. len of db >=1
        assert len(self._db) >=1

        # 8. same freqstr
        assert self._db.index.freqstr == self._future_db.index.freqstr

    def _import_data(self, exogeneous_data : pd.DataFrame, endogeneous_data : pd.Series, endogeneous_var : str) -> tuple:
        '''
        Helper method to iimport the macro and financial data into one Pandas dataframe.

        Keyword arguments:
        exogeneous_data -- as defined in __init__
        endogeneous_data-- as defined in __init__
        endogeneous_var -- as defined in __init__

        Returns:
        A tuple of the form (db, future_db) where db is a pandas DataFrame adhering to the rep invariant for _db
        and future_db is a Pandas DataFrame adhering to the rep invariant for _future_db.  Both are indexed 
        by the period of endogeneous_data

        Raises:
        ValueError if frequency of exogeneous_data is less than the frequency of endogeneous_data, 
            that is, if for any time period in the frequency of endogeneous data between
            the starting period and the last period in which exogeneous_data does not 
            contain an observation.
        IntersectionError if the intersection of the two datasets is the empty set.
        '''
        # get period of endogeneous
        freqstr = endogeneous_data.index.freqstr
        # freqstr can return None if not set.  Should be set, so assert not None
        assert freqstr != None
        if exogeneous_data is not None:
            # get exogeneous_data on same period index
            exogeneous_data = exogeneous_data.resample(freqstr, convention='end').agg('mean')
            exogeneous_data.index = exogeneous_data.index.to_period(freqstr)

            # understand which data is present vs future
            start_period = max(endogeneous_data.index.min(), exogeneous_data.index.min())
            current_period = min(endogeneous_data.index.max(), exogeneous_data.index.max())
            last_period = exogeneous_data.index.max()

            # drop any data after end of exogeneous
            # now dataset only has trailing exog data
            endogeneous_data = endogeneous_data[endogeneous_data.index <= current_period]

            assert last_period >= current_period

            # Merge the DB's
            db = pd.merge(exogeneous_data, endogeneous_data, how='outer', left_index=True, right_index=True)

            # could just drop null, since either exog or endog will be null in at least one col before min date
            # this is more sfb because we will not drop any potential bad nulls in intersection

        else:
            db = endogeneous_data.to_frame()
            start_period = endogeneous_data.index.min()
            current_period = endogeneous_data.index.max()
            last_period = current_period

        # drop data before start period
        db = db.loc[db.index >= start_period]

        # frequency error could only happen if:
        #   - null values in exogeneous data
        #   - missing period observations
        assert not np.any(db.index.duplicated())
        time_delta = db.index.max() - db.index.min()
        exogeneous_cols = list(db.columns)
        exogeneous_cols.remove(endogeneous_var)
        if np.any(db.loc[:,exogeneous_cols].isnull()) or len(db.index) != time_delta.n +1:
            raise ValueError('Frequency of exogeneous data less than frequency of endogeneous data')
        
        future_data = db.loc[db.index > current_period]
        realized_data = db.loc[db.index <= current_period]
        future_data = future_data.drop(endogeneous_var, axis = 1)

        if not len(realized_data) >=1:
            raise IntersectionError('The intersection between exogeneous and endogeneous data is the empty set')

        return (realized_data, future_data)

    # Attributes
    @property
    def exogeneous_vars(self) -> list:
        '''All exogeneous variable names'''
        # TODO tuple would be more SFB
        return deepcopy(self._exogeneous_vars)

    @property
    def endogeneous_data(self) -> pd.Series:
        '''Endogeneous data in float64 format indexed by PeriodIndex.'''
        return pd.Series(self._db.loc[:, self.endogeneous_var], dtype='float64').copy()

    @property
    def exogeneous_data(self) -> pd.DataFrame:
        '''Exogeneous data in float64 format indexed by PeriodIndex.'''
        return self._db.loc[:, self._exogeneous_vars].copy()

    @property
    def future_exogeneous_data(self) -> pd.DataFrame:
        return self._future_db.copy()

    # Observers
    def plot_endogeneous(self) -> pd.Series:
        '''Plots the endogeneous var.'''
        # TODO add title and axes labels
        return self._db.loc[:,self.endogeneous_var].plot(title=self.endogeneous_var, xlabel='Time period')
    
    def plot_autocorrelation(self):
        '''Creates an autocorrelation plot for the endogeneous var.'''
        # TODO fix wrong time periods on axes
        return pdplot.autocorrelation_plot(self._db.loc[:,self.endogeneous_var])
    
    # def get_exogeneous_data(self, vars=None) -> pd.DataFrame:
    #     '''
    #     Getter method for exogeneous data.

    #     If no exogeneous variables, returns a DataFrame with the time series index
    #         but no columns.

    #     Keyword arguments:
    #     vars (optional) -- An iterable subset of exogeneous variables to return.  Default None, which returns all exogeneous columns.
        
    #     Raises:
    #     KeyError if any of cols is not an exogeneous var.
    #     '''
    #     # can't get endog data from this function
    #     # raise DeprecationWarning('get_exogeneous_data will be eliminated in a future version.  Use exogeneous_data instead.')
    #     return_cols = vars if vars else self._exogeneous_vars
    #     if self.endogeneous_var in return_cols:
    #         raise KeyError('endogeneous_var is not exogeneous')

    #     return self._db.loc[:, return_cols].copy()

    def get_future_exogeneous(self, horizon : int=None, vars : int=None) -> pd.DataFrame:
        '''
        Getter method for future exogeneous data, 
        that is, exogeneous data whose corresponding endogenous data value
        has not been realized yet.

        Keyword arguments:
        horizon (optional): the number of future periods to return.  Default None, which returns all future periods.
        vars (optional) -- An iterable subset of exogeneous vars to return.  Default None, which returns all exogeneous vars.

        Returns: 
        The future exogenous data in the form of a DataFrame with a PeriodIndex of the same frequency
        as endogeneous_data.

        Raises:
        ValueError if horizon is greater than available exogeneous data
        KeyError if any of vars are not in the data.
        '''
        # raise DeprecationWarning('get_future_exogeneous will be eliminated in a future version.  Use future_exogeneous_data instead.')
        return_cols = vars if vars else self._exogeneous_vars

        if horizon:
            if horizon > len(self._future_db):
                raise ValueError('Not enough data for horizon ' + str(horizon))
            return self._future_db.loc[:, return_cols].iloc[:horizon].copy()
        else:
            return self._future_db.loc[:, return_cols].copy()

    def _fit_pca(self, explained_variance : float) -> PCA:
        '''
        Helper method to fit the exogeneous data to a principal components analysis (PCA).
        Fits a PCA to the data using the minimum number of variables required to explain
        explained_variance amount of the data.

        Keyword arguments:
        explained_variance: the amount of variance to be explained by the selected principal components.
            Requires 0. <= explained_variance <= 1.0

        Returns: 
        An SKLearn PCA object fitted to the data.
        '''
        # Scale the exogeneous vars to mean 0 and variance 1
        exog_data = self._db.loc[:, self._exogeneous_vars]
        transformed_input = StandardScaler().fit_transform(exog_data)

        # PCA doesnt deal with float 1.0, starts to mean n_components
        # must deal with this case
        if 0.0 < explained_variance < 1.0:
            n_components = explained_variance
        elif not explained_variance < 1.0: # have already checked that its not greater than 1.0
            n_components = min(exog_data.shape) # min # observations, # vars
        elif not explained_variance >0.0:
            n_components = 0
        else:
            raise ValueError('Should never get here')
        # PCA doesnt work with float 0.0, need to convert to int 0 for 0 components

        # Apply PCA to scaled vars
        pca = PCA(n_components=n_components)
        pca = pca.fit(transformed_input)

        return pca

    def get_PCA_covariance_matrix(self, explained_variance : float=0.9) -> pd.DataFrame:
        '''
        Gets a covariance matrix for a data transformation.  

        Gets the covariance matrix between the exogeneous varaibles and the 
        minimum number of principal components necessary to explain explained_variance 
        amount of the data.

        Keyword arguments:
        explained_variance -- float; the amount of variance to be explained by the selected principal components.
            Requires 0 <= explained_variance <= 1.  Default 0.9
            If 0, no explained variance so all exogeneous variables are dropped.
            If 1, the number of components equals the min of the number of observations and the 
                number of exogeneous variables.

        Returns:
        A pandas dataframe of dtype float64, where each row represents a unique exogeneous var and each column represents
        a unique principal component of the minimum set of principal components to explain explained_variance
        amount of the data.  Column names reflect the exogeneous variable that principal component is most
        correlated with.
        '''
        assert 0 <= explained_variance <= 1

        if len(self.exogeneous_vars) > 0:
            exog_data = self._db.loc[:, self._exogeneous_vars]

            pca = self._fit_pca(explained_variance)
            
            # Find which vars correlate most heavily to PCA components
            # DB where columns are PCA components, rows are original exogeneous vars
            # values represent absolute value of correlation between them
            components_db = pd.DataFrame(pca.components_, columns=exog_data.columns).transpose()
            # need absolute value for max correlation
            absolute_value_components_db = components_db.abs()
            # for each PCA var (col) get id (row/exog var) that has highest val
            vars_correlated_pca = absolute_value_components_db.idxmax(axis=0)
            # get value of highest correlation
            # values_correlated_pca = componentsDB.max(axis=0)
            # values_correlated_pca.index = vars_correlated_pca
            col_names = ['PCA' + str(i) + '_correlated_w_' + str(list(vars_correlated_pca)[i]) for i in range(len(vars_correlated_pca))]
            
            components_db.columns = col_names

            self._checkrep()
            return components_db
        else:
            return pd.DataFrame(dtype='float64')

    # Producers

    def downsample(self, exogeneous_vars : list) -> TimeSeriesData:
        '''
        Downsample the exogeneous variables included in the dataset to those in exogeneous_vars.

        Keyword arguments:
        exogeneous_vars -- the exogeneous vars to include in the new dataset.

        Returns:
        A new TimeSeries object containing the same endogeneous and exogeneous data as self,
            but with only exogeneous_vars kept.

        Raises:
        KeyError if any one of exogeneous_vars is not found in the current dataset.
        '''
        self._checkrep()
        assert self.endogeneous_var not in exogeneous_vars, 'Endogeneous var is not exogeneous'
        cols = list(exogeneous_vars)
        cols.append(self.endogeneous_var)
        for col in exogeneous_vars:
            if col not in self._db.columns:
                raise KeyError('Exogeneous var ' + str(col) + 'not found in data')
        return self._from_db(self._db.reindex(columns=cols), self._future_db.reindex(columns=exogeneous_vars), self.endogeneous_var)

    def PCA(self, explained_variance : float=0.9) -> TimeSeriesData:
        '''
        Transforms the exogeneous data using PCA.

        Keyword arguments:
        explained_variance -- float; the amount of variance to be explained by the selected principal components.
            Requires 0 <= explained_variance <= 1.  Default 0.9
            If 0, no explained variance so all exogeneous variables are dropped.
            If 1, the number of components equals the min of the number of observations and the 
                number of exogeneous variables.

        Returns: 
        A new time series dataset ontaining the transformed data.
        The transformed columns are the lowest number of principal compnents that explain at least explained_variance
        of the data.  Column labels are based on the original variable that is most correlated with that
        principal component.
        '''
        assert 0.0 <= explained_variance <= 1.0, 'Explained variance' + str(explained_variance) + 'does not satisfy 0 <= explained variance <= 1'

        # if not 0 cols, then at least one col and one row in exog
        # only thing to check is less than 1 row in future, which we do below
        # otherwise transformation should work
        if len(self._exogeneous_vars) > 0:
            exog_data = self._db.loc[:, self._exogeneous_vars]
            # get PCA object
            pca = self._fit_pca(explained_variance)
            exog_data_pca = pca.transform(exog_data)

            # PCA transform doesnt accept empty dataset
            if len(self._future_db) > 0:
                future_exog_data_pca = pca.transform(self._future_db)
            else:
                future_exog_data_pca = self._future_db

            # columns of db returned by get_PCA_covariance_matrix are named by 
            # exog var that is most correlated, which is what we want here
            col_names = self.get_PCA_covariance_matrix(explained_variance).columns

            # Put in pandas frame
            exog_data_pca = pd.DataFrame(exog_data_pca, columns=col_names)
            future_exog_data_pca = pd.DataFrame(future_exog_data_pca, columns=col_names)
            exog_data_pca.index = self._db.index
            future_exog_data_pca.index = self._future_db.index

            endog_data = self._db.loc[:, self.endogeneous_var]

            pca_db = pd.concat((exog_data_pca, endog_data), axis=1)

            self._checkrep()
            return TimeSeriesData._from_db(pca_db, future_exog_data_pca, self.endogeneous_var)
        else:
            # Ok to return self since immutable
            return self

    def ADF_test(self) -> int:
        '''
        Performs an ADF test on whether the endogeneous variable of the time series is stationary.

        Returns:
        The order of recommended differencing as an integer between 0 and 2.
        '''
        return arima.ndiffs(self._db.loc[:,self.endogeneous_var], test='adf')

    def KPSS_test(self) -> int:
        '''
        Performs a KPSS test for whether the endogeneous variable of the time series is stationary.

        Returns:
        The order of recommended differencing as an integer between 0 and 2.
        '''
        return arima.ndiffs(self._db.loc[:,self.endogeneous_var], test='kpss')
