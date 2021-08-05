from src.data import *
from tests.test_utils import TestUtils
import unittest
import pandas as pd
import numpy as np
import datetime as dt

# warnings.showwarning = warn_with_traceback

class TestData(TestUtils, unittest.TestCase):
    '''Tests for data class'''

    # Testing strategy:
    # Partition on __init__:
    #   exogeneous_data len: None, 1, >1
    #   # exogeneous vars: 0, 1, >1
    #   endogeneous_data len: 1, >1
    #   # future periods: 0, 1, >1
    #   # exog obs/ period: 0, 1, >1
    #   exog start: before endog, after endog, None
    #   exog end: before endog, after endog, None
    #   endogeneous var name len: 0, 1, >1
    #   raises: ValueError, None
    #   IntersectionError: raises, none
    # Partition on exogenous_data:
    #   len: 1, >1
    #   # vars: 1, >1
    # Partition on endogeneous_data:
    #   len: 1, >1
    # Partition on future_exogeneous_data:
    #   len: 0, 1 , >1
    #   # vars: 0, 1, >1
    # Partition on exogeneous_vars:
    #   # cols: 0, 1, >1
    # Partition on downsample:
    #   len exogeneous_vars: 0, 1, >1
    #   KeyError: Raises, None
    # Partition on PCA and get_PCA_covariance_matrix
    #   explained_variance: 0, 0<explained_variance<1, 1
    #   # exogeneous_vars: 0, 1, >1
    #   # exogeneous observations: 1, >1
    #   # exog vs # observations: exog>obs, exog=obs, exog<obs
    # ADF_test -- no need to test since one liner
    # KPSS_test -- no need to test since one liner


    # def compare_data_helper(self, data : TimeSeriesData, expected_exogeneous, expected_endogeneous, expected_future_exogeneous):
    #     result_exogeneous = data.exogeneous_data
    #     result_endogeneous = data.endogeneous_data
    #     result_future_exogeneous = data.get_future_exogeneous()

    #     self.assert_equal_dataframes(expected_exogeneous, result_exogeneous)
    #     self.assert_equal_dataframes(expected_endogeneous, result_endogeneous)
    #     self.assert_equal_dataframes(expected_future_exogeneous, result_future_exogeneous)

    # Covers:
    # Partition on __init__
    #   exogeneous_data_len = None, # exogeneous vars = 0, endogeneous data len = 1
    #   # futue periods = 0, exogeneous obs/period = 0, exog start = None, exog end = None,
    #   endogeneous var name len >1, ValueError = None, IntersectionError = None
    # Partition on exogeneous_data:
    #   len = 1, # vars = 0
    # Partition on endogeneous data:
    #   len = 1
    # Partition on futue_exogeneous_data
    #   cols = None, horizon = None, len = 0, # vars = 0, ValueError = None
    # Partition on exogeneous_vars:
    #   # cols = 0
    # Partition on downsample:
    #   len endogeneous_var = 1, KeyError = Raises
    # Partition on PCA and get_PCA_covariance_matrix:
    #   explained variance = 0<explained_variance<1, # exogeneous = 0, # observations >1
    #   exog vs obs = exog < obs
    def test_data_simple(self):
        endogeneous_index = pd.PeriodIndex(freq='M', year=[2009], month=[1])
        exogeneous_data = None
        endogeneous_var = 'endogeneous'
        endogeneous_data = pd.Series(data=np.array([1.1]),
                                        index=endogeneous_index,
                                        dtype='float64',
                                        name=endogeneous_var)
        
        expected_exogeneous = pd.DataFrame(
                                            index=endogeneous_index,
                                            dtype='float64',
                                            columns=[])
        expected_endogeneous = endogeneous_data
        expected_future_exogeneous = pd.DataFrame(
                                        index=pd.PeriodIndex(data=np.array([]), freq='M'),
                                        dtype='float64',
                                        columns=[])
        expected_exogeneous_cols = set()

        original_data = TimeSeriesData(exogeneous_data, endogeneous_data, endogeneous_var)

        explained_variance = 0.5
        for data in [original_data, original_data.PCA(explained_variance)]:
            result_exogeneous = data.exogeneous_data
            result_endogeneous = data.endogeneous_data
            result_future_exogeneous = data.get_future_exogeneous()
            result_exogeneous_cols = data.exogeneous_vars
            result_cov_matrix = data.get_PCA_covariance_matrix(explained_variance)

            self.assert_equal_dataframes(expected_exogeneous, result_exogeneous)
            self.assert_equal_dataframes(expected_endogeneous, result_endogeneous)
            self.assert_equal_dataframes(expected_future_exogeneous, result_future_exogeneous)
            self.assertEqual(expected_exogeneous_cols, set(result_exogeneous_cols))
            self.assertEqual(len(expected_exogeneous_cols), len(result_exogeneous_cols))
            self.assertEqual((0,0), result_cov_matrix.shape)

            with self.assertRaises(KeyError):
                data.downsample(['your mom'])

    # Covers:
    # Partitions on __init__:
    #   exogeneous data len = 1, # exogeneous vars >1, endogeneous data len >1
    #   # future periods = 1, # exog obs/period = 0, exog start = after endog, exog end = after endog,
    #   endogeneous var name len = 0, ValueError = None, IntersectionError = raises
    def test_data_empty_merge(self):
        endogeneous_index = pd.PeriodIndex(freq='Q',
                                            year=[1999, 1999],
                                            quarter=[2,3])
        exogeneous_index = pd.DatetimeIndex(data=np.array([dt.datetime(1999, 10, 1)]))
        exogeneous_data = pd.DataFrame(data=np.array([[1.1, 2.2]]),
                                            index=exogeneous_index,
                                            dtype='float64',
                                            columns=['var1', 'var2'])
        endogeneous_var = ''
        endogeneous_data = pd.Series(data=np.array([-1, -2]),
                                        index=endogeneous_index,
                                        dtype='float64',
                                        name=endogeneous_var)

        with self.assertRaises(IntersectionError):
            TimeSeriesData(exogeneous_data, endogeneous_data, endogeneous_var)

    # Covers:
    # Partition on __init__:
    #   exogeneous data len > 1, # exogeneous vars = 1, endogeneous_data len >1,
    #   # future periods >1, # exog obs/period = 1, exog start = before endog, exog end = after endog,
    #   endogeneous_var name len = 1, ValueError = None, IntersectionError = None
    # Partition on exogeneous_data:
    #   cols = specified, len >1, # vars = 1
    # Partition on endogeneous_data:
    #   len >1
    # Partiton on future_exogeneous_data:
    #   cols = specified, horizon >1, =1, len >1, # vars = 1, ValueError = None
    # Partition on exogeneous_vars:
    #   # cols = 1
    # Partition on downsample:
    #   len exogeneous_vars = 0, KeyError = None
    # Partition on PCA and get_PCA_covariance_matrix:
    #   explained_variance = 0, # exogeneous = 1, # observations >1, # exogeneous vs # observatins = exog < obs
    def test_data_future(self):
        endogeneous_index = pd.PeriodIndex(freq='M', 
                                            year=[2021, 2021],
                                            month=[6,7])
        exogeneous_index = pd.DatetimeIndex(data=np.array([dt.datetime(2021,5,31),
                                                            dt.datetime(2021,6,1),
                                                            dt.datetime(2021,7,15),
                                                            dt.datetime(2021,8,1),
                                                            dt.datetime(2021, 9,30)]))
        exogeneous_cols = ['myVar']
        exogeneous_data = pd.DataFrame(data=np.array([1,2,3,4,5]),
                                        index=exogeneous_index,
                                        dtype='float64',
                                        columns=exogeneous_cols)
        endogeneous_var = 'v'
        endogeneous_data = pd.Series(data=np.array([-2, -3]),
                                    index=endogeneous_index,
                                    dtype='float64',
                                    name=endogeneous_var)

        past_index = pd.PeriodIndex(freq='M', year=[2021,2021], month=[6,7])
        expected_exogeneous = pd.DataFrame(data=np.array([2,3]),
                                            index=past_index,
                                            dtype='float64',
                                            columns=exogeneous_cols)
        expected_endogeneous = pd.Series(data=np.array([-2,-3]),
                                            index=past_index,
                                            dtype='float64',
                                            name=endogeneous_var)
        future_index_multiple = pd.PeriodIndex(freq='M',
                            year=[2021,2021],
                            month=[8,9])
        expected_future_exogeneous_multiple = pd.DataFrame(data=np.array([4,5]),
                                            index=future_index_multiple,                    
                                            dtype='float64',
                                            columns=exogeneous_cols)
        expected_future_exogeneous_single = pd.DataFrame(data=np.array([4]),
                                            index=pd.PeriodIndex(freq='M',
                                                                    year=[2021],
                                                                    month=[8]),
                                            dtype='float64',
                                            columns=exogeneous_cols)
        data = TimeSeriesData(exogeneous_data, endogeneous_data, endogeneous_var)
        result_exogeneous = data.exogeneous_data
        result_endogeneous = data.endogeneous_data
        result_future_exogeneous_multiple = data.get_future_exogeneous(vars=['myVar'])
        result_future_exogeneous_single = data.get_future_exogeneous(horizon=1)
        result_exogeneous_cols = data.exogeneous_vars

        self.assert_equal_dataframes(expected_exogeneous, result_exogeneous)
        self.assert_equal_dataframes(expected_endogeneous, result_endogeneous)
        self.assert_equal_dataframes(expected_future_exogeneous_multiple, result_future_exogeneous_multiple)
        self.assert_equal_dataframes(expected_future_exogeneous_single, result_future_exogeneous_single)
        self.assertEqual(set(exogeneous_cols), set(result_exogeneous_cols))
        self.assertEqual(len(set(exogeneous_cols)), len(result_exogeneous_cols))

        # downsample and PCA
        # downsample to 0 cols, PCA 0 which results in 0 cols
        downsample_exogeneous_cols = []
        expected_downsample_exogeneous = pd.DataFrame(
                                            index=past_index,
                                            dtype='float64',
                                            columns=downsample_exogeneous_cols)
        expected_downsample_future_exogeneous = pd.DataFrame(
                                                    index=future_index_multiple,
                                                    dtype='float64',
                                                    columns=downsample_exogeneous_cols)
        explained_variance = 0.0
        for downsample_data in [data.downsample(downsample_exogeneous_cols), data.PCA(explained_variance)]:
            self.assert_equal_dataframes(expected_downsample_exogeneous, downsample_data.exogeneous_data)
            self.assert_equal_dataframes(expected_endogeneous, downsample_data.endogeneous_data)
            self.assert_equal_dataframes(expected_downsample_future_exogeneous, downsample_data.get_future_exogeneous())
            self.assertEqual(set(downsample_exogeneous_cols), set(downsample_data.exogeneous_vars))
            self.assertEqual(len(set(downsample_exogeneous_cols)), len(downsample_data.exogeneous_vars))
            self.assertEqual((0,0), downsample_data.get_PCA_covariance_matrix(explained_variance).shape)
        

    # Covers:
    # Partition on __init__:
    #   exogeneous_data len >1, # exogeneous vars >1, endogeneous_data len >1,
    #   # future periods = 0, # exog obs/ period >1, exog start = before endog, exog end: before endog,
    #   endogeneous var name len >1, ValueError = None, IntersectionError = None
    # Partition on exogenous_data:
    #   len = 1, # vars >1
    # Partition on endogeneous_data:
    #   len = 1
    # Partition on future_exogeneous_data:
    #   cols = None, horizon = 1, 0, len = 0, # vars >1, raises = ValueError
    # Partition on exogeneous_vars:
    #   # cols >1
    # Partition on downsample:
    #   len exogeneous_vars >1, KeyError = None
    # Partition on PCA and get_PCA_covariance_matrix:
    #   explained_variance = 1, # exogeneous >1, # obs = 1, exog vs obs = exog>obs
    def test_data_average_exogeneous(self):
        endogeneous_index = pd.PeriodIndex(freq='M', 
                                            year=[2021, 2021],
                                            month=[6,7])
        exogeneous_index = pd.DatetimeIndex(data=np.array([dt.datetime(2021,5,1),
                                                            dt.datetime(2021,5,31),
                                                            dt.datetime(2021,6,1),
                                                            dt.datetime(2021,6,30)]))
        exogeneous_cols = ['var1', 'var2']
        exogeneous_data = pd.DataFrame(data=np.array([[17, 18],[34,36], [1, 2],[-1, -2]]),
                                        index=exogeneous_index,
                                        dtype='float64',
                                        columns=exogeneous_cols)
        endogeneous_var = 'endog'
        endogeneous_data = pd.Series(data=np.array([1, 2]),
                                    index=endogeneous_index,
                                    dtype='float64',
                                    name=endogeneous_var)

        past_index = pd.PeriodIndex(freq='M', year=[2021], month=[6])
        expected_exogeneous = pd.DataFrame(data=np.array([[0,0]]),
                                            index=past_index,
                                            dtype='float64',
                                            columns=exogeneous_cols)
        expected_endogeneous = pd.Series(data=np.array([1]),
                                            index=past_index,
                                            dtype='float64',
                                            name=endogeneous_var)
        expected_future_exogeneous = pd.DataFrame(
                                            index=pd.PeriodIndex(freq='M', data=np.array([])),
                                            dtype='float64',
                                            columns=exogeneous_cols)
        original_data = TimeSeriesData(exogeneous_data, endogeneous_data, endogeneous_var)
        for data in (original_data, original_data.downsample(exogeneous_cols)):
            result_exogeneous = data.exogeneous_data
            result_endogeneous = data.endogeneous_data
            with self.assertRaises(ValueError):
                data.get_future_exogeneous(horizon=1)
            result_future_exogeneous = data.get_future_exogeneous()
            result_exogeneous_cols = data.exogeneous_vars

            self.assert_equal_dataframes(expected_exogeneous, result_exogeneous)
            self.assert_equal_dataframes(expected_endogeneous, result_endogeneous)
            self.assert_equal_dataframes(expected_future_exogeneous, result_future_exogeneous)
            
            self.assertEqual(set(exogeneous_cols), set(result_exogeneous_cols))
            self.assertEqual(len(set(exogeneous_cols)), len(result_exogeneous_cols))
        
        explained_variance = 1.0
        # Test PCA: Resulting dataframe should only have 1 col
        # spec not strong enough to determine what value col takes
        data_PCA = original_data.PCA(explained_variance)
        self.assertEqual((1,1), data_PCA.exogeneous_data.shape)
        self.assertEqual((1,), data_PCA.endogeneous_data.shape)
        self.assertEqual((0,1), data_PCA.get_future_exogeneous().shape)
        # column names change, but still should be same num and no duplicates
        self.assertEqual(1, len(data_PCA.exogeneous_vars))
        self.assertEqual(1, len(set(data_PCA.exogeneous_vars)))

        pca_cov_matrix = original_data.get_PCA_covariance_matrix(explained_variance)
        self.assertEqual((2,1), pca_cov_matrix.shape)
        # each col (exogenoues var) is now a row in the PCA cov matrix
        self.assertEqual(set(original_data.exogeneous_vars), set(pca_cov_matrix.index))
        self.assertFalse(np.any(pca_cov_matrix.index.isnull()))
        self.assertFalse(np.any(pca_cov_matrix.isnull()))
        self.assertFalse(np.any(np.isinf(pca_cov_matrix)))

    # Covers:
    # Partition on __init__
    #   exogeneous_data len >1, # exogeneous_vars = 1, endogeneous_data len = 1
    #   # future periods >1, # exogeneous obs/period = 0, exog start = before endog, exog end = after endog,
    #   endogeneous_var name len = 0, ValueError = raises, IntersectionError = None
    def test_inadequate_frequency(self):
        endogeneous_index = pd.PeriodIndex(freq='Q',
                                            year=[1999],
                                            quarter=[3])
        exogeneous_index = pd.DatetimeIndex(data=np.array([dt.datetime(1999, 9, 30),
                                                        dt.datetime(2000,1,1)]))
        exogeneous_data = pd.DataFrame(data=np.array([1.1, 2.2]),
                                            index=exogeneous_index,
                                            dtype='float64',
                                            columns=['var1'])
        endogeneous_var = ''
        endogeneous_data = pd.Series(data=np.array([-1]),
                                        index=endogeneous_index,
                                        dtype='float64',
                                        name=endogeneous_var)

        with self.assertRaises(ValueError):
            TimeSeriesData(exogeneous_data, endogeneous_data, endogeneous_var)