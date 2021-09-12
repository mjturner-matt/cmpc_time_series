import argparse
import sys
from ui.ui_utils import make_model_object
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

    parser.add_argument('endogeneous_file', type=str, help='Filepath to endogeneous data file.  Raises OSError if path does not exist.'
                                                + 'File format must comply with one of the formats provided in README.')
    parser.add_argument('endogeneous_format', type=time_series.EndogeneousDataFormats, choices=list(time_series.EndogeneousDataFormats), help='Format for endogeneous_file.')
    parser.add_argument('endogeneous_var', type=str, help='Name of the endogeneous var.')
    parser.add_argument('--exogeneous_file', type=str, help='Filepath to exogeneous data file. Raises OSError if path does not exist.'
                                                + 'File format must comply with one of the formats provided in README.'
                                                + 'Raises ArgumentTypeError if provided but no exogeneous_format specified.')
    parser.add_argument('--exogeneous_format', type=time_series.ExogeneousDataFormats, choices=list(time_series.ExogeneousDataFormats), help='Format for exogeneous_file. '  
                                                + 'Raises ArgumentTypeError if speciffied but no exogeneous_file provided.')
    parser.add_argument('--pca', nargs='?', type=float, const=0.8, default=None, help='Proportion of variance explained by optional principal components analysis.  Requires 0 <= pca <= 1')
    parser.add_argument('--model', type=time_series.TimeSeriesRegressionModels, choices=time_series.TimeSeriesRegressionModels, default=time_series.TimeSeriesRegressionModels.SARIMARegression, help='The time series model to use.')
    parser.add_argument('--p_d_q_order', nargs='+', default=[0,0,0], type=int, help='Order arguments for the model.')
    parser.add_argument('--P_D_Q_order', nargs='+', default=[0,0,0,4], type=int, help='Seasonal order arguments for the model.')
    parser.add_argument('--fit', action='store_true', default=False, help='Fit model and print results.')
    parser.add_argument('--predict', nargs='?', const=0, default=None, help='Horizon for future predictions.')
    parser.add_argument('--sliding_window', nargs='?', type=float, const=0.8, default=None, help='Training window size for sliding window forecast.  Requires 0.0 <= window_size <= 1.0.')
    parser.add_argument('--outfile', type=str, default=sys.stdout, help='Directory to save files. Raises OSError if path doesn\'t exist.' 
                                                + 'Saves files in folder according to naming convention \{option\}.csv, i.e. pca.csv, predict.csv')

    args = parser.parse_args(sys_args)

    # Check preconditions on args
    # files exist
    if not path.isfile(args.endogeneous_file):
        raise OSError('Path to endogeneous_file does not exist.')
    if args.exogeneous_file and not path.isfile(args.exogeneous_file):
        raise OSError('Path to exogeneous_file does not exist.')
    if args.outfile != sys.stdout and not path.isdir(args.outfile):
        raise OSError('Path to outfile does not exist')

    if args.outfile != sys.stdout:
        pca_outfile = args.outfile + 'pca.csv'
        predict_outfile = args.outfile + 'predict.csv'
        sliding_window_outfile = args.outfile + 'sliding_window.csv'
    else:
        pca_outfile = sys.stdout
        predict_outfile = sys.stdout
        sliding_window_outfile = sys.stdout

    # format specified
    if bool(args.exogeneous_file) != bool(args.exogeneous_format):
        raise argparse.ArgumentTypeError('xogeneous_file must be specified if and only if exogeneous_format is specified.')

    if args.pca:
        if not 0.0 <= args.pca <= 1.0:
            raise argparse.ArgumentTypeError('pca must be between 0.0 and 1.0')
    
    for order in args.p_d_q_order + args.P_D_Q_order:
        if not order >= 0:
            raise argparse.ArgumentTypeError('Order args must be >= 0.')
    
    if args.sliding_window:
        if not 0.0 <= args.sliding_window <= 1.0:
            raise argparse.ArgumentTypeError('Sliding_window must be between 0.0 and 1.0')

    endog_raw = time_series.EndogeneousDataImporter.import_endogeneous(args.endogeneous_file, args.endogeneous_var, args.endogeneous_format)
    if args.exogeneous_file:
        exog_raw = time_series.ExogeneousDataImporter.import_exogeneous(args.exogeneous_file, args.exogeneous_format)
    else:
        exog_raw = None

    raw_data = time_series.TimeSeriesData(exog_raw, endog_raw, args.endogeneous_var)

    # None if no, const if spec without arg
    if args.pca:
        to_csv(raw_data.get_PCA_covariance_matrix(args.pca), pca_outfile)
        data = raw_data.PCA(args.pca)
    else:
        data = raw_data
    
    endog_data = data.endogeneous_data
    exog_data = data.exogeneous_data
    future_exog = data.get_future_exogeneous()
    arima_model = make_model_object(args.model, args.p_d_q_order, args.P_D_Q_order, endog_data, exog_data)

    if args.fit:
        print(arima_model.fit(endog_data, exog_data))

    if args.predict != None:
        # TODO not sure what get_future_exogeneous does if no future exogeneous data
        # need to spec better and test
        to_csv(arima_model.predict(args.predict, endog_data, exog_data, future_exog), predict_outfile)
    
    if args.sliding_window:
        # TODO make saving fig optional
        to_csv(arima_model.sliding_window_forecast(endog_data, exog_data, args.sliding_window), sliding_window_outfile)

