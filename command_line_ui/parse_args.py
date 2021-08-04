import argparse
from src import time_series
from src.utils import *
from os import path

def main(sys_args):
    '''
    Parses input arguments and runs the corresponding program.

    Keyword arguments:
    args -- 
    '''
    parser = argparse.ArgumentParser(description="Tools to import data and run a time series regression.")

    # converters for choices available
    # define here, parser enforces choices as keys of these dicts
    model_converters = {'brute': time_series.BruteForceARIMARegression,
                    'auto': time_series.AutoARIMARegression,
                    'sarima' : time_series.SARIMARegression}

    output_converters = {'to_excel' : to_excel,
                    'plot' : plot_dataframe}

    parser.add_argument('macro_file', type=str, help='filepath to excel file containing microeconomic data. Raises IOError if path does not exist.'
                                                + 'File format must comply with all preconditions documented in CMPCFinancialMacroData')
    parser.add_argument('financial_statement_file', type=str, help='filepath to csv file containing CMPC financial data.  Raises IOError if path does not exist.'
                                                + 'File format must comply with all preconditions documented in CMPCFinancialMacroData')
    parser.add_argument('financial_format', type=time_series.EndogeneousDataFormats, choices=list(time_series.EndogeneousDataFormats), help='The format for the financial statement_file.')
    parser.add_argument('endogeneous_var', type=str, help='row name from CMPC financial data file to use as endogeneous var in the regression.'
                                                            + 'Must be in financial_statement_file and have non-null values for all time periods specified.')
    parser.add_argument('--pca', type=float, default=False, help='proportion of variance explained by optional PCA analysis.  Requires 0 <= pca <= 1')
    # TODO check filepath
    parser.add_argument('--save_components_covariance_matrix', default=False, help='filepath to save covariance matrix for the principal components')
    parser.add_argument('--model', type=str, default='auto', choices=model_converters.keys(), help='the time series model to use')
    #TODO default order shouldn't be 0
    # TODO fail fast on bad orders
    parser.add_argument('--p_d_q_order', nargs='+', default=(2,1,2), type=int, help='the order arguments for the model')
    parser.add_argument('--P_D_Q_order', nargs='+', default=(1,0,1,4), type=int, help='the seasonal order arguments for the model')
    parser.add_argument('--fit', action='store_true', default=False, help='fit regression to exogeneous data')
    parser.add_argument('--predict', action='store_true', default=False, help='predict all future time periods')
    parser.add_argument('--sliding_window', action='store_true', default=False, help='use a sliding window forecast to evaluate model prediction accuracy')
    # TODO fail fast here
    parser.add_argument('--window_size', type=float, default=0.8, help='the fraction of data to use for window_size.  Requires 0 <= window_size <= 1.'
                                                                    + 'Raises ParseError if --rolling_window not specified.')
    parser.add_argument('--sliding_window_out', type=str, default='plot', choices=output_converters.keys(), help='option for sliding window results'
                                                                    + 'Raises ParseError if --rolling_window not specified.')
    parser.add_argument('--out_filename', default='rolling_window_output', type=str, help='output fielpath for rolling window results'
                                                                    + 'Raises ParseError if --rolling_window not specified.')

    args = parser.parse_args(sys_args)

    # Check path validity - fail fast
    if not path.exists(args.macro_file):
        raise OSError('Filepath to macro_file does not exist')
    if not path.exists(args.financial_statement_file):
        raise OSError('Filepath to financial_statement_file does not exist')

    exog_raw = time_series.ExogeneousDataImporter.import_exogeneous(args.macro_file, time_series.ExogeneousDataFormats.macro)
    endog_raw = time_series.EndogeneousDataImporter.import_endogeneous(args.financial_statement_file, args.endogeneous_var, args.financial_format)
    data = time_series.TimeSeriesData(exog_raw, endog_raw, args.endogeneous_var)
    if args.save_components_covariance_matrix:
        # TODO what if pca 0.9 not specified
        to_excel(data.get_PCA_covariance_matrix(args.pca), args.save_components_covariance_matrix)

    # prepare data for model
    cmpc_endog = data.endogeneous_data
    if args.pca:
        # TODO what if want db but not pca in model
        pca_data = data.PCA()
        cmpc_exog = pca_data.get_exogeneous_data()
        cmpc_future_exog = pca_data.get_future_exogeneous()
    else:
        cmpc_exog = data.get_exogeneous_data()
        cmpc_future_exog = data.get_future_exogeneous()

    if args.model == 'brute':
        # TODO spec that orders correspond to max ranges here
        # order_arg_ranges = (range(0,4), #p - Cannot equal m
        #     range(0,3), #d
        #     range(0,4), #q - Cannot equal m
        #     range(0,2), #P
        #     range(0,2), #D
        #     range(0,2)) #Q
        p, d, q = args.p_d_q_order
        P, D, Q, m = args.P_D_Q_order
        order_arg_ranges = (range(0, p+1),
                range(0, d+1),
                range(0, q+1),
                range(0, P+1),
                range(0, D+1),
                range(0, Q+1),
                range(m, m+1))
        arima_model = model_converters[args.model](cmpc_endog, cmpc_exog, order_arg_ranges)
    elif args.model == 'auto':
        pdq = args.p_d_q_order if '--p_d_q_order' in sys_args else None
        PDQ = args.P_D_Q_order if '--P_D_Q_order' in sys_args else None
        arima_model = model_converters[args.model](pdq, PDQ)
    else:
        arima_model = model_converters[args.model](tuple(args.p_d_q_order), tuple(args.P_D_Q_order))

    if args.fit:
        print(arima_model.fit(cmpc_endog, cmpc_exog))

    if args.predict:
        # TODO make predict work when no exog (can't take length of empty)
        to_excel(arima_model.predict(len(cmpc_future_exog), cmpc_endog,cmpc_exog, cmpc_future_exog), args.out_filename)
    
    if args.sliding_window:
        # TODO make saving fig optional
        output_converters[args.sliding_window_out](arima_model.sliding_window_forecast(cmpc_endog, cmpc_exog, args.window_size), args.out_filename)
    else:
        # check that no other rolling args specified
        if '--window_size' in sys_args or '--sliding_window_out' in sys_args:
            # TODO probably shouldn't be putting None in 
            raise argparse.ArgumentError(None, 'sliding window options given but --sliding_window is not specified')
    
    if '--out_filename' in sys_args and not args.predict and not args.sliding_window:
        raise argparse.ArgumentError(None, 'out_file option given but neither --sliding_window nor --predict is specified')
