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

def range_combo_generator(ranges):
    '''
    Generates all combinations of integers within each range in ranges.

    Keyword arguments:
    ranges -- An iterable of range objects.
    
    Yields:
    A tuple of len(ranges) containing the next combination of integers,
        where the ith element of the tuple is within the ith range of ranges.
    '''
    def range_combo_helper(i, ranges):
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

def calc_rmse(yhats, actuals):
    '''Calculates the rmse between yhats and actuals'''
    return sqrt(mean_squared_error(actuals, yhats))

def plot_dataframe(dataframe, filename):
    '''Plots a line plot of the dataframe columns and saves to filename'''
    dataframe.plot()
    # plt.show()
    plt.savefig(filename)

def to_excel(dataframe, filename):
    '''Saves the dataframe to an excel file with name filename'''
    dataframe.to_excel(filename)