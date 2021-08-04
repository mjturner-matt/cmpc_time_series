from datetime import time
import unittest
from command_line_app import parse_args
from argparse import ArgumentError
from os import path
import os


class TestMain(unittest.TestCase):

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

    # TODO refactor to one helper method to check for absence of file, run code, check for presence, then delete
    def check_file_created(function, args, filename):
        raise NotImplementedError


    financial_statement_file = 'tests/cmpc_income_q_corrected_quarter.csv'
    macro_file = 'tests/data.xlsx'

    # Covers:
    #   macro_file: absent, financial_statement_file: absent
    def test_absent_files(self):

        no_macro = ['file_doesnt_exist.xlsx', self.financial_statement_file, 'cmpc_financial_data','Profit, before taxes']
        with self.assertRaises(OSError):
            parse_args.main(no_macro)

        no_financial = [self.macro_file, 'file_doesnt_exist', 'cmpc_financial_data', 'Profit, before taxes']
        with self.assertRaises(OSError):
            parse_args.main(no_financial)

    # Covers:
    #   macro_file: present, financial_statement_file: present
    #   pca: specified, save_matrix: default, model: default, p_d_q_order: default, P_D_Q_order: default
    #   fit: specified, predict: default, sliding_window: default, window_size: default, sliding_window_out: plot,
    #   out_filename: deffault, result: error
    def test_fit_invalid_args(self):

        # do PCA so model isn't overspecified
        args = [self.macro_file, self.financial_statement_file, 'cmpc_financial_data', 'Profit, before taxes', '--pca', '0.7', '--fit', '--sliding_window_out', 'plot']
        with self.assertRaises(ArgumentError):
            parse_args.main(args)

    # Covers
    #   macro_file: present, financial_statement_file: present
    #   pca: float, save_matrix: default, model: brute, p_d_q_order: specified, P_D_Q_order: specified,
    #   fit: default, predict: default, sliding_window: specified, window_size: specified, sliding_window_out: default, 
    #   out_filename: specified, result: correct
    def test_brute_correct(self):
        out_filename = 'tests/test_brute_correct.png'
        # make sure file isnt there previously
        if path.exists(out_filename):
            os.remove(out_filename)
        args = [self.macro_file, self.financial_statement_file, 'cmpc_financial_data', 'Profit, before taxes', '--pca', '0.9',
                '--model', 'brute', '--p_d_q_order', '1', '0', '0', '--P_D_Q_order', '1', '0', '0', '4',
                '--sliding_window', '--out_filename', out_filename]
        
        parse_args.main(args)
        self.assertTrue(path.exists(out_filename))
        # delete file created
        os.remove(out_filename)
        self.assertFalse(path.exists(out_filename))

    # Covers:
    #   macro_file: present, financial_statement_file: present,
    #   pca: default, save_components_covariance_matrix: filename
    def test_save_pca_covariance(self):
        cov_matrix_filepath = 'tests/test_save_pca_covariance.xlsx'
        # verify doesnt exist already
        if path.exists(cov_matrix_filepath):
            os.remove(cov_matrix_filepath)

        args = [self.macro_file, self.financial_statement_file, 'cmpc_financial_data', 'Profit, before taxes',
                            '--save_components_covariance_matrix', cov_matrix_filepath]
        
        parse_args.main(args)

        self.assertTrue(path.exists(cov_matrix_filepath))
        
        os.remove(cov_matrix_filepath)
        self.assertFalse(path.exists(cov_matrix_filepath))

    # Covers:
    #   macro_file: present, financial_statement_file: present,
    #   pca: float, save_components_covariance_matrix: default, model: auto, p_d_q_order: default, P_D_Q_order: default
    #   fit: specified, predict: default, sliding_window: default, window:size: default, sliding_window_out: default, out_filename: default,
    #   result: correct
    def test_auto_fit(self):
        args = [self.macro_file, self.financial_statement_file, 'cmpc_financial_data', 'Profit, before taxes', 
                '--pca', '0.7', '--model', 'auto', '--fit']

        parse_args.main(args)
        # weak spec, only care that no error is thrown
        self.assertTrue(True)

    # Covers:
    #   macro_file: present, financial_statment_file: present, 
    #   pca: float, save_components_covariance_matrix: default, 
    #   model: sarima, p_d_q_order: default, P_D_Q_order: default,
    #   fit: default, predict: default, sliding_window: specified, window_size: default, 
    #   sliding_window_out: to_excel, out_filename: present, result: correct
    def test_sliding_window_to_excel(self):
        out_filename = 'tests/test_sliding_window_to_excel.xlsx'

        args = [self.macro_file, self.financial_statement_file, 'cmpc_financial_data', 'Profit, before taxes', 
                '--pca', '0.7', '--model', 'sarima', '--sliding_window', '--sliding_window_out',
                 'to_excel', '--out_filename', out_filename]

        # assert not in path before
        if path.exists(out_filename):
            os.remove(out_filename)

        parse_args.main(args)

        self.assertTrue(path.exists(out_filename))
        os.remove(out_filename)
        self.assertFalse(path.exists(out_filename))

    # Covers:
    #   macro_file: present, financial_statement_file: present,
    #   pca: float, save_components_covariance_matrix: default, model: auto, sarima, p_d_q_order: default, P_D_Q_order: default
    #   fit: default, predict: default, sliding_window: default, window_size: default, sliding_window_out: default, out_filename: default,
    #   result: correct
    def test_predict(self):
        for model in ['auto', 'sarima']:
            out_filename = 'tests/test_predict.xlsx'

            args = [self.macro_file, self.financial_statement_file, 'cmpc_financial_data', 'Profit, before taxes', 
                    '--pca', '0.9', '--model', model, '--predict', '--out_filename', out_filename]

            if path.exists(out_filename):
                os.remove(out_filename)

            parse_args.main(args)

            self.assertTrue(path.exists(out_filename))
            os.remove(out_filename)
            self.assertFalse(path.exists(out_filename))


if __name__ == '__main__':
    unittest.main()
