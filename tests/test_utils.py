import unittest
import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    '''Print traceback for all warnings'''
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

class TestUtils():

    def assert_equal_dataframes(self, expected, result):
        '''
        Asserts two dataframes are equal in:
        - Elements (including order)
        - Value and frequency of indices
        - dtype
        '''
        self.assertTrue(expected.equals(result))
        self.assertTrue(expected.index.equals(result.index))
        self.assertEqual(expected.index.freqstr, result.index.freqstr)

    def assert_equal_series(self, expected, result):
        '''
        Asserts two series are equal in:
        - Elements (including order)
        - Value and frequency of indices
        - dtype
        '''
        self.assert_equal_dataframes(expected, result)