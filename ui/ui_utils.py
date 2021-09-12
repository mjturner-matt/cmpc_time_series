from src import time_series
import pandas as pd

def make_model_object(type : time_series.TimeSeriesRegressionModels, p_d_q_order : tuple, P_D_Q_order : tuple, y : pd.Series, X : pd.DataFrame) -> time_series.SARIMARegression:
    '''
    Makes a time series model according to model type.

    Parameters
    ----------
    type
        Model type.
    p_d_q_order
        The order of the model.
    P_D_Q_order
        The seasonal order of the model.
    y
        Endogeneous data.
    X 
        Exogeneous data.

    Returns
    -------
    A subclass of SARIMARegression.
    '''
    p, d, q = p_d_q_order
    P, D, Q, m = P_D_Q_order

    if type == time_series.TimeSeriesRegressionModels.BruteForceARIMARegression:
        order_arg_ranges = (range(0, p+1),
            range(0, d+1),
            range(0, q+1),
            range(0, P+1),
            range(0, D+1),
            range(0, Q+1),
            range(m, m+1))
        ts = getattr(time_series, type.value)(y, X, order_arg_ranges)
    else:
        ts = getattr(time_series, type.value)((p, d, q), (P, D, Q, m))
    
    return ts