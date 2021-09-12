from ast import parse
from datetime import time
from tests.test_utils import TestUtils
import unittest
from unittest import mock
from unittest.mock import MagicMock, call, patch

from command_line_app import parse_args
from argparse import ArgumentTypeError
from os import path
import os
import sys

import builtins
from src.utils import make_future_index
from src.time_series import SARIMARegression

import pandas as pd
import numpy as np

def pathExists(filepath):
    return path.exists(filepath)

class TestMain(unittest.TestCase, TestUtils):

    # TODO test for presence of test files

    # Testing strategy:
    # Automatic partititons on main
    #   macro_file: present, absent
    #   financial_statement file: present, absent
    #   endogeneous_var
    #   pca: default, float
    #   save_matrix: default, false, filename
    #   model: default, brute, auto, arima
    #   p_d_q_order: default, specified
    #   P_D_Q_order: default, specified
    #   fit: default, specified
    #   predict: default, specified
    #   sliding_window: default, specified
    #   window_size: default, specified
    #   sliding_window_out: default, plot, to_excel
    #   out_filename: default, specified
    #   result: correct, error
    # Manual tests:
    #   TODO specify manual tests

    # New testing strategy
    # Testing strategy:
    # Partitions on main:
    #   endog file loc: valid filepath, OSError
    #   endog format: cmpc_financial_data, excel_table
    #   exog_file: present, absent
    #   exog_file_loc: valid filepath, OSError, None
    #   exog_format: macro, absent
    #   pca: value, present, absent
    #   model: sarima, auto, brute, absent
    #   p_d_q_order: present, absent
    #   P_D_Q_order: present, absent
    #   fit: present, absent
    #   predict: value, present, absent
    #   sliding_window: value, present, absent
    #   outfile: present, absent
    #   outfile loc: valid filepath, stdout, OSError, None
    #   OSError: Raises, None
    #   ArgumentTypeError: Raises, None

    def assertPathExists(self, filepath):
        if not pathExists(filepath):
            raise AssertionError("File does not exist: %s" % str(filepath))

    def assertPathNotExists(self, filepath):
        if pathExists(filepath):
            raise AssertionError("File does not exist: %s" % str(filepath))
    
    # TODO refactor to one helper method to check for absence of file, run code, check for presence, then delete
    def check_file_created(function, args, filename):
        raise NotImplementedError


    financial_statement_file = 'tests/cmpc_income_q_corrected_quarter.csv'
    macro_file = 'tests/data.xlsx'

    # NEW
    test_file_path = 'tests/parse_args_test_cases/'
    out_file_path = test_file_path + 'out_tmp/'
    invalid_out_file_path = 'invalid_out/'
    cmpc_financial_data_file = test_file_path + 'cmpc_financial_data.csv'
    excel_table_file = test_file_path + 'excel_table.xlsx'
    macro_file = test_file_path + 'macro.xlsx'
    predict_out_file = out_file_path + 'predict.csv'
    sliding_window_out_file = out_file_path + 'sliding_window.csv'
    pca_out_file = out_file_path + 'pca.csv'

    cmpc_financial_data_endogeneous_var = 'myVar'

    excel_table_index = pd.period_range(start=pd.Period(year=2000, freq='M'), periods=12, freq='M')
    excel_table_endogeneous_var = 'EBITDA'
    excel_table_y = pd.Series(data=np.arange(start=1.0, stop=13.0),
                                index=excel_table_index,
                                name=excel_table_endogeneous_var,
                                dtype='float64')
    excel_table_x = pd.DataFrame(
                                index=excel_table_index,
                                columns=[],
                                dtype='float64')
    excel_table_n_predictions = 1
    excel_table_predictions = pd.Series(data=np.random.rand(excel_table_n_predictions),
                                        index=make_future_index(excel_table_index, excel_table_n_predictions),
                                        dtype='float64',
                                        name='Predictions')
    excel_table_empty_predictions = pd.Series(
                                        index=make_future_index(excel_table_index, 0),
                                        dtype='float64',
                                        name='Predictions')
    excel_table_sliding_window = pd.DataFrame(data=np.array([[],[]]).T,
                                        index=pd.PeriodIndex(data=np.array([]), 
                                                        freq='M'),
                                        columns=['predictions', 'actuals'],
                                        dtype='float64')

    def setUp(self) -> None:
        self.assertPathNotExists(self.predict_out_file)
        self.assertPathNotExists(self.sliding_window_out_file)
        self.assertPathNotExists(self.invalid_out_file_path)
        self.assertPathNotExists(self.pca_out_file)

    def tearDown(self) -> None:
        if pathExists(self.predict_out_file):
            os.remove(self.predict_out_file)
        if pathExists(self.sliding_window_out_file):
            os.remove(self.sliding_window_out_file)
        if pathExists(self.invalid_out_file_path):
            os.remove(self.invalid_out_file_path)
        if pathExists(self.pca_out_file):
            os.remove(self.pca_out_file)

    # Covers:
    #   endog file loc = OSError, endog format = excel_table,
    #   exog file = absent, exog_file loc = None, exog_format = absent
    #   pca = absent, model = absent, p_d_q_order = absent, P_D_Q_order = absent
    #   fit = absent, predict = absent, sliding_window = absent
    #   outfile = absent, outfile loc = None
    #   OSError = Raises, ArgumentTypeError = None
    def test_absent_endog_file(self):
        absent_filename = self.test_file_path + 'idontexist'
        self.assertPathNotExists(absent_filename)

        args = [absent_filename, 'excel_table', 'myVar']
        with self.assertRaises(OSError):
            parse_args.main(args)

    # Covers:
    #   endog file loc = valid filepath, endog format = cmpc_financial_data,
    #   exog file = present, exog_file loc = OSError, exog_format = macro,
    #   pca = absent, model = absent, p_d_q_order = absent, P_D_Q_order = absent
    #   fit = absent, predict = absent, sliding_window = absent
    #   outfile = absent, outfile loc = None
    #   OSError = Raises, ArgumentTypeError = None
    def test_absent_exog_file(self):
        endog_filename = self.test_file_path + 'test_absent_exog_file.csv'
        exog_absent_filename = self.test_file_path + 'nanana'
        self.assertPathExists(endog_filename)
        self.assertPathNotExists(exog_absent_filename)

        args = [endog_filename, 'cmpc_financial_data', 'myVar', '--exogeneous_file', exog_absent_filename, '--exogeneous_format', 'macro']
        with self.assertRaises(OSError):
            parse_args.main(args)

    # Covers:
    #   endog file loc = valid filepath, endog format = excel_table,
    #   exog file = absent, exog_file loc = None, exog_format = macro
    #   pca = absent, model = absent, p_d_q_order = absent, P_D_Q_order = absent
    #   fit = absent, predict = absent, sliding_window = absent
    #   outfile = absent, outfile loc = None
    #   OSError = None, ArgumentTypeError = Raises
    def test_parse_invalid_args(self):
        args = [self.excel_table_file, 'excel_table', 'myVar', '--exogeneous_file', self.macro_file]
        args2 = [self.excel_table_file, 'excel_table', 'myVar', '--exogeneous_format', 'macro']

        with self.assertRaises(ArgumentTypeError):
            parse_args.main(args)
        
        with self.assertRaises(ArgumentTypeError):
            parse_args.main(args2)

    # Covers:
    #   endog file loc = valid filepath, endog format = excel_table,
    #   exog file = absent, exog_file loc = None, exog_format = absent
    #   pca = absent, model = brute, p_d_q_order = absent, P_D_Q_order = absent
    #   fit = present, predict = absent, sliding_window = absent
    #   outfile = absent, outfile loc = None
    #   OSError = None, ArgumentTypeError = None
    @patch('builtins.print')
    @patch('src.time_series.BruteForceARIMARegression.determine_opt_order')
    @patch('src.time_series.BruteForceARIMARegression.fit')
    def test_simple_brute_fit(self, mock_ts, mock_opt_order, mock_print):
        # Setup
        args = [self.excel_table_file, 'excel_table', 'EBITDA',
                    '--model', 'BruteForceARIMARegression', '--fit']

        mock_ts.return_value = "fit return"
        mock_opt_order.return_value = (0,0,0,0,0,0,4)

        parse_args.main(args)

        mock_ts.assert_called_once()
        y, X = mock_ts.call_args.args

        self.assert_equal_series(y, self.excel_table_y)
        self.assert_equal_dataframes(X, self.excel_table_x)

        mock_print.assert_called_once_with(mock_ts.return_value)

    # Covers:
    #   endog file loc = valid filepath, endog format = cmpc_financial_data,
    #   exog file = present, exog_file loc = valid filepath, exog_format = macro
    #   pca = present, model = absent, p_d_q_order = absent, P_D_Q_order = absent
    #   fit = absent, predict = absent, sliding_window = absent
    #   outfile = present, outfile loc = valid filepath
    #   OSError = None, ArgumentTypeError = None
    def test_save_pca_covariance(self):
        cov_matrix_filepath = self.out_file_path + 'pca.csv'
        # verify doesnt exist already

        args = [self.cmpc_financial_data_file, 'cmpc_financial_data', self.cmpc_financial_data_endogeneous_var,
        '--exogeneous_file', self.macro_file, '--exogeneous_format', 'macro',
                            '--pca', '--outfile', self.out_file_path]
        
        parse_args.main(args)

        self.assertTrue(path.exists(cov_matrix_filepath))
        
        os.remove(cov_matrix_filepath)
        self.assertFalse(path.exists(cov_matrix_filepath))

    # Covers:
    #   endog file loc = valid filepath, endog format = excel_table,
    #   exog file = absent, exog_file loc = None, exog_format = absent
    #   pca = value, model = auto, p_d_q_order = present, P_D_Q_order = present
    #   fit = absent, predict = value, sliding_window = present
    #   outfile = present, outfile loc = valid filepath
    #   OSError = None, ArgumentTypeError = None
    @patch('src.time_series.AutoARIMARegression.sliding_window_forecast')
    @patch('src.time_series.AutoARIMARegression.predict')
    def test_auto_predict_sliding(self, mock_predict : MagicMock, mock_sliding : MagicMock):
        args = [self.excel_table_file, 'excel_table', self.excel_table_endogeneous_var,
                '--pca', '0.8', '--model', 'AutoARIMARegression', '--p_d_q_order', '1', '0', '0', '--P_D_Q_order', '0', '0', '0', '4',
                '--predict', '1', '--sliding_window', 
                '--outfile', self.out_file_path]

        mock_predict.return_value = self.excel_table_predictions
        mock_sliding.return_value = self.excel_table_sliding_window

        parse_args.main(args)

        mock_predict.assert_called_once()
        mock_sliding.assert_called_once()

        self.assertPathExists(self.predict_out_file)
        self.assertPathExists(self.sliding_window_out_file)


    # Covers:
    #   endog file loc = valid filepath, endog format = excel_table,
    #   exog file = absent, exog_file loc = None, exog_format = absent
    #   pca = absent, model = sarima, p_d_q_order = absent, P_D_Q_order = absent
    #   fit = absent, predict = value, sliding_window = present
    #   outfile = absent, outfile loc = stdout
    #   OSError = None, ArgumentTypeError = None
    @patch('src.time_series.SARIMARegression.sliding_window_forecast')  
    @patch('src.time_series.SARIMARegression.predict')
    def test_sarima_stdout(self, mock_predict : MagicMock, mock_sliding_window : MagicMock):
        args = [self.excel_table_file, 'excel_table', self.excel_table_endogeneous_var,
                '--model', 'SARIMARegression',
                '--predict', '--sliding_window', '1.0']

        expected_predictions = self.excel_table_empty_predictions
        expected_sliding_window = self.excel_table_sliding_window

        mock_predict.return_value = expected_predictions
        mock_sliding_window.return_value = expected_sliding_window

        with patch.object(sys.stdout, 'write') as mock_stdout:
            parse_args.main(args)

        mock_predict.assert_called_once()
        mock_sliding_window.assert_called_once()

        call_args = mock_stdout.call_args_list

        self.assert_equal_series(expected_predictions, call_args[0][0][0])
        self.assert_equal_series(expected_sliding_window, call_args[1][0][0])

    # Covers:
    #   endog file loc = valid filepath, endog format = cmpc_financial_data
    #   exog file = absent, exog_file loc = None, exog_format = absent
    #   pca = absent, model = absent, p_d_q_order = absent, P_D_Q_order = absent
    #   fit = absent, predict = absent, sliding_window = absent
    #   outfile = present, outfile loc = OSError
    #   OSError = Raises, ArgumentTypeError = None
    def test_invalid_outfile(self):
        args = [self.cmpc_financial_data_file, 'cmpc_financial_data', self.cmpc_financial_data_endogeneous_var,
            '--outfile', self.invalid_out_file_path]

        with self.assertRaises(OSError):
            parse_args.main(args)


if __name__ == '__main__':
    unittest.main()
