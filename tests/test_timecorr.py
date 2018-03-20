from past.utils import old_div
import numpy as np
import pandas as pd
import hypertools as hyp
import scipy.spatial.distance as sd
from scipy.linalg import toeplitz

from timecorr.timecorr import timecorr, levelup
import timecorr.helpers 

gaussian_params = {'var': 1000}
data_list= np.random.randn(10,3)
pandas_dataframe= pd.DataFrame(np.random.randint(low=0, high=10, size=(2, 2)))
#pandas_dataframes= pd.df()
numpy_array= np.array([[5, 9], [10, 7]])
numpy_array_list= np.array([[8,2],[4,6]]).tolist()
random_numbers= (2 ,3 ,5, 10, 12, 4, 6)


def test_timecorr():
    data_df = hyp.tools.format_data(pandas_dataframe)
    data_npa = hyp.tools.format_data(numpy_array)
    data_npl = hyp.tools.format_data(numpy_array_list)
    data_rn = hyp.tools.format_data(random_numbers)
    assert isinstance(data_df, np.array)
    assert isinstance(data_npa, np.array)
    assert isinstance(data_npl, np.array)
    assert isinstance(data_rn, np.array)

    first_tester = gaussian_weights(T, params=gaussian_params)
    assert isinstance (first_tester, )


   # test_weights_functions=
    assert isinstance(first_tester, np.array)

def test_levelup ():
    data = hyp.tools.format_data(data_list)
    if type(data) == list:
        V = data[0].shape[1]
    else:
        V = data.shape[1]

    corrs = timecorr(data, weights_function=weight_function, weights_params=weights_params, mode="within", cfun=isfc)
    return hyp.reduce(corrs, reduce=reduce, ndims=V)
    assert isinstance(corrs.shape == data.shape)
