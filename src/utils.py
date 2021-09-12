from __future__ import annotations

import sys

import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

from matplotlib import pyplot as plt

# display options
pd.set_option("display.max_columns", 999)
pd.set_option("display.max_rows", 999)

'''utils.py
Utilities for src.
'''

def range_combo_generator(ranges : tuple) -> tuple:
    '''
    Generates all combinations of integers within each range in ranges.

    Keyword arguments:
    ranges -- An iterable of range objects.
    
    Yields:
    A tuple of len(ranges) containing the next combination of integers,
        where the ith element of the tuple is within the ith range of ranges.
    '''
    def range_combo_helper(i : int, ranges : tuple) -> tuple:
        '''
        Helper method that yields all combinations of integers from the range objects
        located in ranges[i:].

        Keyword arguments:
        i -- integer index in [0, len(ranges))
        ranges -- An iterable where ranges[i:] are all range objects.

        Yields:
        A tuple of len(ranges) where indices [i:] are the next combination of integers from
        the range objects present in ranges[i:]
        '''
        # Base case: index is length of ranges, so all elements of ranges are now integers
        if i == len(ranges):
            yield ranges
        else: # recursive step: for each integer in the range at the current index
            # make all combinations beyond that index
            for j in ranges[i]:
                yield from range_combo_helper(i+1, ranges[0:i] + (j,) + ranges[i+1:])
    yield from range_combo_helper(0, ranges)

def calc_rmse(yhats : pd.Series, actuals : pd.Series) -> float:
    '''Calculates the rmse between yhats and actuals'''
    return sqrt(mean_squared_error(actuals, yhats))

def plot_dataframe(dataframe : pd.DataFrame, filename : str):
    '''Plots a line plot of the dataframe columns and saves to filename'''
    dataframe.plot()
    # plt.show()
    plt.savefig(filename)

def to_excel(dataframe : pd.Series | pd.DataFrame, filename : str):
    '''Saves the data to an excel file with name filename'''
    dataframe.to_excel(filename)

def to_csv(dataframe : pd.Series | pd.DataFrame, filename : str):
    '''Saves the data to a csv file with name filename.'''
    if filename == sys.stdout:
        sys.stdout.write(dataframe)
    else:
        dataframe.to_csv(filename)

def sliding_window_rmse(sliding_window_results : pd.DataFrame) -> float:
    '''
    Calculates the RMSE for sliding window results.

    Keyword arguments:
    sliding_window_results: the results dataframe of sliding_window_forecast

    Returns:
    The rmse of the results.
    '''
    return calc_rmse(sliding_window_results['predictions'], sliding_window_results['actuals'])

def make_future_index(index : pd.PeriodIndex, n : int) -> pd.PeriodIndex:
    '''
    Makes a future PeriodIndex of length n.

    Future index is of the same frequency as index, begins exactly one period after
    the maximum period of index, and contains n periods with no gaps or duplicate
    periods.

    Parameters
    ----------
    index : pd.PeriodIndex
        The current index.  Must be of length >0.
    n : int
        Number of future periods to extend.

    Returns
    -------
    pd.PeriodIndex
        New index of length n of the same frequency as index beginning at the period
        directly following index.
    '''
    start_date = index.max() + index.freq
    return pd.period_range(start=start_date, periods=n, freq=index.freq)