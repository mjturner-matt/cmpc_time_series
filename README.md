# cmpc_financial_time_series
Time series analysis of CMPC financial statements.

# inputs
The package accepts the following input formats for both endogeneous and exogeneous data.

Endogeneous data:

excel_table:
    - xlsx file of only one sheet
    - Cell 1A must contain the term "Year"
    - Column A contains the year of the observation(s) in YYYY format.
        Must contain year label for every nonempty row.
    - Row 1 contains the month of the observations.  May be in any format recognized by dateparser library.
        Not case sensitive.  Must contain month label for every nonempty column.
    - There must be exactly one observation per month between the first and last months given, inclusive.

cmpc_financial_data:
    - csv file
    - "Description" appears in row 1.  It's column contains the names of all variables,
        including endogeneous_var.  It's row contains the quarter for each column of observations
            in the format QX YY.  All other columns not in that format will be dropped.
    - There is exactly 1 column per quarter.  There is exactly one column per quarter for each
        quarter between the min and max quarters specified.
    - endogeneous_var must appear in the column 'Description' exactly once.  It's row contains the value for the endogeneous var.
    - endogeneous_var cannot be the empty string
    - '', '-', ' ' are treated as null values, in addition to NaN's.  All nulls in quarterly columns
        are filled with 0.0's.
    - Blank rows are skipped
    - Negative numbers are represented with -, as in -123.45.  () for negative numbers may NOT be used.
        Commas cannot be used to break apart large numbers (ie 123,456 is not allowed).  

Exogeneous data:
macro:
    - xlsx file
    - Data must be located in the only table on a sheet named 'BBG P'
    - Row 4 must contain unique column names, each representing a unique variable.
    - Rows 1, 2, 3, 5, and 6 are extraneous header information and will be skipped upon import.
        Row 7 contains the first row of actual data.  Any rows with a missing index in column A are skipped.
    - Cell 4A must be empty or null.
    - Column A contains the date index for the data in '%m/%d/%y' format.
    - Data begins in column B.
    - Null values or empty strings are ok in data-containing cells, but will be backfilled 
        with the least recent non-null cell.  If a column is missing the most recent observation and cannot
        be backfilled, it will be forward filled.  Completely empty columns are dropped.