from os import path
import pandas as pd
import numpy as np
import dateparser
import datetime as dt
from enum import Enum

'''Tools to import both endogeneous and exogeneous data based on known file types.

For each type of data (endogeneous and exogeneous) there is an importer class containing
a single factory method to import data of a particular format.  Valid formats are 
contained in an enum and their specifications can be found in the Readme for this package.

Typical usage examples:
foo = EndogeneousDataImporter.import_endogeneous(filename, endogeneous_var, EndogeneousDataFormats.format_name)
bar = ExogeneousDataImporter.import_exogeneous(filename, ExogeneousDataFormats.format_name)
'''

class EndogeneousDataFormats(Enum):
    '''
    Enumerates allowed file formats for endogeneous data import.
    '''
    cmpc_financial_data = 'cmpc_financial_data'
    excel_table = 'excel_table'

class ExogeneousDataFormats(Enum):
    '''
    Enumerates allowed file formats for exogeneous data import.
    '''
    macro = 'macro'

class EndogeneousDataImporter:
    '''Import endogeneous data'''

    def import_endogeneous(file : str, endogeneous_var : str, format : EndogeneousDataFormats) -> pd.Series:
        '''
        Factory method for importing time indexed endogeneous data.

        Keyword arugments:
        file -- file-like or file path; the endogeneous data.
            Requires the to file conforms to format as specified in the Readme.
        endogeneous_var -- The name of the endogeneous var.
        format -- The format of the endogeneous data.

        Returns:
        A series of endogeneous_var of type float64, indexed by period and sorted in increasing date order, 
        where no entry is null, no period is duplicated, and no intermediate period is missing.
        '''
        if format == EndogeneousDataFormats.cmpc_financial_data:
            func = import_cmpc_financial_data
        elif format == EndogeneousDataFormats.excel_table:
            func = import_excel_table
        else:
            raise ValueError('Endogeneous data format not recognized')

        data = func(file, endogeneous_var)
        EndogeneousDataImporter._check_import(data)
        return data

    def _check_import(data):
        '''Asserts the postcondition of import_endogeneous.'''
        # if shape is len 1, 1-dimensional array, only 1 col
        assert len(data.shape) == 1
        assert not np.any(data.isnull())

        # TODO refactor in utility method to DRY
        assert not np.any(data.index.duplicated())
        # Every quarterly observation between max and min filled
        # if length 0 index, then satisfied since no max or min
        # can't test since .max() of empty index is pd.NaT
        if not len(data.index) == 0:
            time_delta = data.index.max() - data.index.min()
            # should be a +1 on time delta since we have 40 periods, but observation at beginning of first period
            # example: quarter 1-quarter 0 =1, although we have 2 quarters
            assert len(data.index) == time_delta.n +1

def import_excel_table(filename : str, endogeneous_var : str) -> pd.Series:
    '''
    Converts an excel table to a DataFrame.  
    See spec for EndogeneousDataImporter.import_endogeneous()
    '''
    # import macro vars
    endog_db = pd.read_excel(filename, parse_dates=False)
    # endog_db = endog_db.rename(columns = {"Unnamed: 0": "Year"})
    # print(endog_db)
    # Melt defaults to using all columns for unpivot
    endog_db = endog_db.melt(id_vars=['Year'], var_name='Month', value_name=endogeneous_var)

    endog_db = endog_db.astype({'Year' : 'str', 'Month' : 'str'})
    endog_db.loc[:,'Month'] = endog_db.loc[:, 'Month'].str.upper()

    endog_db.loc[:, 'Date'] = endog_db.loc[:,'Year'] + endog_db.loc[:,'Month']
    # could parse period instead in future version
    # TODO is last the right thing to do?
    endog_db.loc[:, 'Date'] = endog_db.loc[:, 'Date'].apply(lambda x: dateparser.parse(x, settings={'PREFER_DAY_OF_MONTH' : 'last'}))
    endog_db = endog_db.set_index("Date")
    endog_db.index = pd.to_datetime(endog_db.index)
    endog_db = endog_db.loc[:,endogeneous_var]
    endog_db = endog_db.dropna()
    endog_db.index = endog_db.index.to_period('M')
    endog_db = endog_db.astype('float64')
    endog_db = endog_db.sort_index()
    
    return endog_db

def parse_quarter_to_period(quarter : str) -> pd.Period:
    '''
    Converts a string of the form QX YY to period.
    
    Keyword arguments:
    quarter -- String of the format 'QX YY' representing the period quarter X of the most 
        recently ocurring year ending in YY.

    Returns:
    The corresponding period.
    '''
    century = 100 #years
    quarter, year = quarter.split()
    current_year = int(dt.datetime.today().year)
    current_century = (current_year // century) * century
    if int(year) + current_century > current_year:
        four_digit_year = current_century -century + int(year)
    else:
        four_digit_year = current_century + int(year)
    quarter_num = int(quarter[1])
    assert 1 <= quarter_num <= 4
    return pd.Period(freq = 'Q', quarter=quarter_num, year=four_digit_year)


def import_cmpc_financial_data(filename : str, endogeneous_var) -> pd.Series:
    '''
    Converts a csv CMPC Financial Data dataset to a DataFrame.  Designed to work with csv files created
    from the XLSX format available here:
    http://apps.indigotools.com/IR/IAC/?Ticker=CMPC&Exchange=SANTIAGO,
    although an individual csv sheet must be made, and negative numbers must be formatted without parentheses or commas.
        Example: -123456.78 not (123,456.78)

    See spec for EndogeneousDataImporter.import_endogenous()
    '''
    assert endogeneous_var != ''

    financial_statement_db = pd.read_csv(filename, 
                                    parse_dates=True, 
                                    na_values=['', '-', ' '], # endogeneous can't be '' due to treatment as null
                                    skip_blank_lines=True) # many blank rows in CMPC download

    # Massage data into correct format
    # Description is anchor point for col containing endogeneous and row containing quarters
    # index is now vars, including endogeneous, and columns are now quarters plus other columns (ex. 'Currency', 'Unit')
    financial_statement_db = financial_statement_db.set_index('Description')
    # filter columns based on regex
    financial_statement_db = financial_statement_db.filter(regex='Q[1-4]\s\d\d')
    # Columns are now vars, including endogeneous, rows are quarters
    financial_statement_db = financial_statement_db.transpose()

    # Set index
    financial_statement_db.index = financial_statement_db.index.map(parse_quarter_to_period)
    financial_statement_db.index = pd.PeriodIndex(financial_statement_db.index, freq='Q')

    # Drop vars other than endogeneous
    financial_statement_db = financial_statement_db.loc[:,endogeneous_var]
    financial_statement_db = financial_statement_db.fillna(0.0)
    financial_statement_db = financial_statement_db.astype('float64')

    # sort index by time
    financial_statement_db = financial_statement_db.sort_index()

    return financial_statement_db
    
class ExogeneousDataImporter:
    '''Import exogeneous data'''

    def import_exogeneous(file : str, format : ExogeneousDataFormats) -> pd.DataFrame:
        '''
        Factory method to import time indexed exogeneous data.

        Keyword arguments:
        file -- file-like or path; the endogeneous data.
            Requires that the file conforms to format according to the spec for the format found in the Readme.
        format -- The name of the file format.

        Returns:
        A Pandas DataFrame indexed by date where each column contains a unique exogeneous variabe in float64 format.
        The dataframe contains no null values.  Nulls in the imported file will be treated as follows:
        - Nulls occurring prior to existing data will be backfilled.
        - Any nulls between the last date in the index and the most recent observation of that variable will
            be forward filled.
        - Exogeneous variables with no corresponding data (all nulls) will be dropped.
        Data will be sorted in ascending order by date.

        Raises:
        OSError if the filepath does not exist.
        '''
        if format == ExogeneousDataFormats.macro:
            func = import_macro
        else:
            raise ValueError("Exogneous variable format not recognized.")

        data = func(file)
        ExogeneousDataImporter._check_import(data)
        return data

    def _check_import(data):
        '''Asserts the postcondition on import_exogeneous.'''
        # assert no null values after bfill and ffill
        assert not np.any(data.isnull())


def import_macro(filename : str) -> pd.DataFrame:
    '''
    Converts an XLSX macroeconomic dataset to a DataFrame.  
    See spec for factory method ExogeneousDataImporter.import_exogeneous()
    '''
    # import vars
    exog_db = pd.read_excel(filename, sheet_name="BBG P", skiprows=[0,1,2,4,5], index_col = 0, parse_dates=True)
    # change empty string col name back to empty string
    exog_db = exog_db.rename(columns = {"Unnamed: 1": ""})
    # should only have one empty string col name, so if there exists unnamed 1 precondition of unique columns not met
    assert not 'Unnamed: 2' in exog_db.columns, 'Only one column label may be empty string'
    
    # format index
    # drop empty index entries
    exog_db = exog_db[exog_db.index.notnull()]
    exog_db.index = pd.to_datetime(exog_db.index)
    # must be sorted before fill to fill in correct order
    exog_db = exog_db.sort_index()

    # format data
    exog_db = exog_db.astype('float64')

    # fill missing data
    # if only some entries missing, backfill
    exog_db = exog_db.bfill()
    exog_db = exog_db.ffill()

    # after back and forward fill, only empty cols are those entirely empty at beginning
    exog_db = exog_db.dropna(how='all', axis=1)

    # assert unique columns
    cols = set(exog_db.columns)
    assert len(cols) == len(exog_db.columns)
    return exog_db

