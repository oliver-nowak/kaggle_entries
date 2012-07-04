import scipy.io
import scipy.spatial.distance as spsd
import numpy as np
import scipy.stats
import scipy.io

mat = scipy.io.loadmat('./data/kaggle77b_trainset.mat')

test_mat = scipy.io.loadmat('./data/kaggle77b_testset.mat')

row1 = mat['trainset'][0]
row2 = mat['trainset'][1]
row_set = mat['trainset'][:10]


test_row1 = test_mat['testset'][0]
test_row2 = test_mat['testset'][1]


test_array = np.array([row1, test_row1])

def sim_euclidean(u, v):
    """Calculate the Euclidean distance between of two data items (usually rows).

    Returns the normalized distance where 1.0 means the items are exactly the same, and 0.0 are completely different."""
    dist = spsd.euclidean(u, v)
    norm_dist = 1 / (1 + dist)
    print norm_dist

def sim_pearson(x):
    pear_data = np.corrcoef(x)
    print 'saving data...'
    scipy.io.savemat('./data/test_mat.mat', {'pearson_data':pear_data})
    print pear_data
    return pear_data


def clean_data(data, NoDataValue):
    """Replace the NoDataValue with np.nan in order to avoid data skew when we 
    calculate the Pearson correlation"""
    data[data==NoDataValue] = np.nan
    return data

cleaned = clean_data(row1, 99.)
print cleaned
#sim_euclidean(row1, row2)
#sim_pearson(row_set, test_row1)
#print row_set
#print test_row1
#print test_array
#sim_pearson(row_set)
#sim_pearson_r(row_set)
