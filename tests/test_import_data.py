import unittest

from pandas.core.indexes.datetimes import DatetimeIndex
from src.import_data import *
import datetime as dt

class TestEndogeneousDataImporter(unittest.TestCase):

    test_data_folder = 'tests/import_data_test_cases/'

    # Testing strategy:
    # Partitions on import_endogeneous:
    #   endogeneous var len: 0, >0
    #   format: cmpc_financial_data, EXCEL_TABLE
    # import_excel_table:
    #   # years in rows: 0, 1, >1
    #   # months in cols: 0, 1, >1
    #   # years filled: 0, 1, >1
    #   # years filled/rows: <# rows, = # rows
    #   # months filled: 0, 1, >1
    #   months filled/col: < # cols, = # cols
    #   months name: English Spanish
    #   months case: upper lower mixed, None
    #   len endogeneous: 0 >0
    #   # output observations: 0, 1, >1
    # import_cmpc_financial_data
    #   description row: first, middle, last
    #   description loc: before currency/unit, after currency/unit
    #   description loc: before quarter, after quarter
    #   # vars: 1, >1
    #   # quarters: 0, 1, >1
    #   # endog observations: 0, 1, >1
    #   endog observations/quarters: <1, 1
    #   len endogeneous: 1, >1
    #   quarters: sorted, not sorted
    #   # output observations: 0, 1, >1

    def assert_equal_series(self, expected, result):
        '''
        Asserts two series are equal in:
        - Elements (including order)
        - Value and frequency of indices
        - dtype
        '''
        self.assertTrue(expected.equals(result))
        self.assertTrue(expected.index.equals(result.index))
        self.assertEqual(expected.index.freqstr, result.index.freqstr)

    # Covers:
    # Partitions on import_endogeneous:
    #   # endogeneous var len >0, format = EXCEL_TABLE
    # Partitions on import_excel_table:
    #   # years in rows = 0, # months in col = 1, 
    #   # years filled = 0, years filled/rows = # cols, # months filled = 0, months filled/col = <# cols,
    #   months name = English, months case = len endogeneous >0
    #   # output_observations = 0
    def test_no_years_excel(self):
        filename = self.test_data_folder + 'test_no_years.xlsx'
        endogeneous_var = 'bla'

        expected = pd.Series(index=pd.PeriodIndex(np.array([]), freq='M'), name=endogeneous_var, dtype='float64')
        
        result = EndogeneousDataImporter.import_endogeneous(filename, endogeneous_var, EndogeneousDataFormats.excel_table)

        self.assert_equal_series(expected, result)

    # Covers:
    # Partitions on import_endogeneous:
    #   # endogeneous var len =0, format = EXCEL_TABLE
    # Partitions on import_excel_table:
    #   # years in rows = 1, # months in cols = 0,
    #   # years filled=0, years filled/rows = < # rows, # months filled = 0, months filled/cols = # cols
    #   months name = Spanish, months case = None, len endogeneous =0,
    #   # output observations = 0
    def test_no_months_excel(self):
        filename = self.test_data_folder + 'test_no_months.xlsx'
        endogeneous_var = ''

        expected = pd.Series(index = pd.PeriodIndex(np.array([]), freq='M'), name=endogeneous_var, dtype='float64')
        
        result = EndogeneousDataImporter.import_endogeneous(filename, endogeneous_var, EndogeneousDataFormats.excel_table)

        self.assert_equal_series(expected, result)

    # Covers:
    # Partitions on import_endogeneous:
    #   # endogeneous var len >0, format = EXCEL_TABLE
    # Partitions on import_excel_table:
    #   # years in row >1, # months in col >1,
    #   # years filled = 1, years filled/rows < # cols, # months filled = 1, months filled/cols = < # cols
    #   months name = Spanish, months case = lower, len endogeneous >0,
    #   # output observations = 1
    def test_single_cell_excel(self):
        filename = self.test_data_folder + 'test_single_cell.xlsx'
        endogeneous_var = 'hola'
        index = pd.PeriodIndex(freq='M', year=[2015], month=[3])
        expected = pd.Series(data=[1.00001], index = index, name=endogeneous_var)

        result = EndogeneousDataImporter.import_endogeneous(filename, endogeneous_var, EndogeneousDataFormats.excel_table)

        self.assert_equal_series(expected, result)

    # Covers:
    # Partitions on import_endogeneous:
    #   # endogeneous var len =0, format = EXCEL_TABLE
    # Partitions on import_excel_table:
    #   # years in row >1, # months in row >1, 
    #   # years filled >1, years filled/rows < # rows, # months filled >1, months filled/cols < # cols,
    #   # months name = Spanish, months case = lower, len endogeneous = 0,
    #   # output observations >1
    def test_multiple_cells_excel(self):
        filename = self.test_data_folder + 'test_multiple_cells.xlsx'
        endogeneous_var = ''

        expected = pd.Series(data = [-1.1, -2.2], index=pd.PeriodIndex(freq='M', year=[2019, 2020], month=[12, 1]), name=endogeneous_var)

        result = EndogeneousDataImporter.import_endogeneous(filename, endogeneous_var, EndogeneousDataFormats.excel_table)

        self.assert_equal_series(expected, result)

    # Covers:
    # Partitions on import_endogeneous:
    #   # endogeneous var len >0, format = cmpc_financial_data
    # Partitions on import_excel_table:
    #   endog var row = first, description loc = after currency/unit, description loc = before quarter
    #   # vars = 1, # quarters = 0, # endog observations = 0, endog observations/quarters = 1,
    #   len endogeneous = 1, quarters = sorted,
    #   # output observations = 0
    def test_empty_cmpc(self):
        filename = self.test_data_folder + 'test_empty_cmpc.csv'
        endogeneous_var = 'a'

        expected = pd.Series(index = pd.PeriodIndex(data=np.array([]), freq='Q'), name = endogeneous_var, dtype='float64')

        result = EndogeneousDataImporter.import_endogeneous(filename, endogeneous_var, EndogeneousDataFormats.cmpc_financial_data)

        self.assert_equal_series(expected, result)

    # Covers:
    # Partitions on import_endogeneous:
    #   # endogeneous var len >0, format = cmpc_financial_data
    # Partitions on import_excel_table:
    #   endog var row = last, description loc = before currency/unit, description loc = after quarter
    #   # vars >1, # quarters = 1, # endog observations = 1, endog observations/quarters = 1,
    #   len endogeneous >1, quarters = sorted,
    #   # output observations = 1
    def test_two_vars_cmpc(self):
        filename = self.test_data_folder + 'test_two_vars_cmpc.csv'
        endogeneous_var = 'bla'

        expected = pd.Series(data=[-123.45], index=pd.PeriodIndex(freq='Q', year=[1999], quarter=[1]), name=endogeneous_var)

        result = EndogeneousDataImporter.import_endogeneous(filename, endogeneous_var, EndogeneousDataFormats.cmpc_financial_data)

        self.assert_equal_series(expected, result)

    # Covers:
    # Partitions on import_endogeneous:
    #   # endogeneous var len >0, format = cmpc_financial_data
    # Partitions on import_excel_table:
    #   endog var row = middle, description loc = after currency/unit, description loc = before quarter
    #   # vars > 1, # quarters >1, # endog observations >1, endog observations/quarters <1,
    #   len endogeneous >1, quarters = not sorted
    #   # output observations >1
    def test_multi_cmpc(self):
        filename = self.test_data_folder + 'test_multi_cmpc.csv'
        endogeneous_var = 'bla'

        expected = pd.Series(data=[0.0, 0.1, 0.2], index=pd.PeriodIndex(freq='Q', year=[2009, 2009,2010], quarter=[3, 4,1]),name=endogeneous_var)

        result = EndogeneousDataImporter.import_endogeneous(filename, endogeneous_var, EndogeneousDataFormats.cmpc_financial_data)

        self.assert_equal_series(expected, result)

class TestExogeneousDataImporter(unittest.TestCase):

    test_data_folder = 'tests/import_data_test_cases/'

    # Testing strategy:
    # Partitions on import_exogeneous:
    #   format: macro
    # Partitions on import_macro:
    #   # sheets: 1, >1
    #   sheet position: first, middle, last
    #   # vars: 0, 1, >1
    #   # dates: 0, 1, >1
    #   fill: None, backfill, forward fill, drop
    #   dates sorted: ascending, descending, mixed
    #   col names: capital, lower, mixed, none
    #   col name len: 0, >0, none
    #   negatives: yes, no
    #   output rows: 0, 1, >1
    #   output cols: 0, 1, >1

    def assert_equal_dataframes(self, expected, result):
        '''Asserts two dataframes have equal indices and data'''
        self.assertTrue(expected.equals(result))
        self.assertTrue(expected.index.equals(result.index))

    # Covers:
    # Partitions on import_exogeneous:
    #   format = macro
    # Partitions on import_macro:
    #   # sheets >1, sheet position = last, # vars = 0, # dates = 0, fill = None
    #   dates_sorted: col names = None, col name len = none, negatives = no
    #   output rows = 0, output cols = 0
    def test_empty_macro(self):
        filename = self.test_data_folder + 'test_empty_macro.xlsx'

        expected = pd.DataFrame(index=DatetimeIndex(np.array([])), 
                                    columns=[],
                                    dtype='float64')

        result = import_macro(filename)

        self.assert_equal_dataframes(expected, result)

    # Covers:
    # Partitions on import_exogeneous:
    #   format = macro
    # Partitions on import_macro:
    #   # sheets = 1, sheet position = first, # vars = =1, # dates >1, fill = forward
    #   dates sorted = mixed, col names = lower, col names len = 0, negatives = yes,
    #   output rows >1, output cols = 1
    def test_forward_fill_one_var_macro(self):
        filename = self.test_data_folder + 'test_forward_fill_one_var_macro.xlsx'

        expected = pd.DataFrame(np.array([123, -456, -456, -456]), 
            index=pd.DatetimeIndex(data=np.array([dt.datetime(1999,12,31),
                                            dt.datetime(2000,1,1),
                                            dt.datetime(2000,1,4),
                                            dt.datetime(2000,1,6)])),
                                        columns=[''],
                                        dtype='float64')
        
        result = import_macro(filename)

        self.assert_equal_dataframes(expected, result)
    
    # Covers:
    # Partitions on import_exogeneous:
    #   format = macro
    # Partitions on import_macro:
    #   # sheets >1, sheet position = middle, # vars = 1, # dates >1, fill = backfill,
    #   dates sorted = descending, col names = mixed, col names len > 0, negatives = yes,
    #   output rows >1, output cols = 1
    def test_backfill_one_var_macro(self):
        filename = self.test_data_folder + 'test_backfill_one_var_macro.xlsx'

        expected = pd.DataFrame(data=np.array([-1000, -1000, -2000]),
                                    index=pd.DatetimeIndex(data=np.array([dt.datetime(1931,3,1),
                                                                            dt.datetime(1931,3,2),
                                                                            dt.datetime(1931,3,3)]),),
                                    columns=['my_VaR'],
                                    dtype='float64')

        result = import_macro(filename)

        self.assert_equal_dataframes(expected, result)

    # Covers:
    # Partitions on import_exogeneous:
    #   format = macro
    # Partitions on import_macro:
    #   # sheets =1, sheet position = first, # vars >1, # dates = 1, fill = drop,
    #   dates sorted = ascending, col_names = capital, col names length >0, negatives = no,
    #   output rows = 1, output cols >1
    def test_drop_cols_macro(self):
        filename = self.test_data_folder + 'test_drop_cols_macro.xlsx'

        expected = pd.DataFrame(data=np.array([[123], [456]]).T,
                                    index = pd.DatetimeIndex(data = np.array([dt.datetime(2021, 11, 1)])),
                                    dtype='float64',
                                    columns=['B', 'C'])

        result = import_macro(filename)

        self.assert_equal_dataframes(expected, result)

class TestParseQuarterToPeriod(unittest.TestCase):

    # Testing strategy:
    # over parse_quarter_to_period:
    # Test transition between current year and following year number
    #   which should be interpreted as past since it occurs in the future

    def test_year_transition(self):
        current_year = int(dt.datetime.today().year)
        century = 100 # years

        for year in [current_year, current_year + 1 - century]:
            rep_YY = str(year % 100)
        
            rep_Q1 = 'Q1 ' + rep_YY
            rep_Q2 = 'Q2 ' + rep_YY
            rep_Q3 = 'Q3 ' + rep_YY
            rep_Q4 = 'Q4 ' + rep_YY

            expected_Q1 = pd.Period(freq='Q', year=year, quarter=1)
            expected_Q2 = pd.Period(freq='Q', year=year, quarter=2)
            expected_Q3 = pd.Period(freq='Q', year=year, quarter=3)
            expected_Q4 = pd.Period(freq='Q', year=year, quarter=4)

            self.assertEqual(expected_Q1, parse_quarter_to_period(rep_Q1))
            self.assertEqual(expected_Q2, parse_quarter_to_period(rep_Q2))
            self.assertEqual(expected_Q3, parse_quarter_to_period(rep_Q3))
            self.assertEqual(expected_Q4, parse_quarter_to_period(rep_Q4))

