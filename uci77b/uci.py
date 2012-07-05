import scipy.io
import scipy.spatial.distance as spsd
import numpy as np
import scipy.stats
import scipy.io
from math import sqrt

mat = scipy.io.loadmat('./data/kaggle77b_trainset.mat')

test_mat = scipy.io.loadmat('./data/kaggle77b_testset.mat')

row1 = mat['trainset'][0]
row2 = mat['trainset'][1]
row_set = mat['trainset'][:10]


test_row1 = test_mat['testset'][0]
test_row2 = test_mat['testset'][1]


test_array = np.array([row1, test_row1])

train_set = mat['trainset']
test_set = test_mat['testset']
test_row_set = test_mat['testset'][:2]


def sim_euclidean(u, v):
    """Calculate the Euclidean distance between of two data items (usually rows).

    Returns the normalized distance where 1.0 means the items are exactly the same, and 0.0 are completely different."""
    dist = spsd.euclidean(u, v)
    norm_dist = 1 / (1 + dist)
    print norm_dist

def sim_pearson(train_set, test_row, row_num):
    pear_data = np.corrcoef(train_set,test_row)
    file_name = 'pear_data_' + str(row_num).zfill(2) + '.mat'
    print 'saving data for %s...',file_name
    scipy.io.savemat('./data/'+ file_name, {'pearson_data':pear_data})
    print '... %s  data saved.',file_name
    return pear_data
    

def clean_data(data, NoDataValue):
    """Replace the NoDataValue with np.nan in order to avoid data skew when we 
    calculate the Pearson correlation"""
#   data[data==NoDataValue] = np.nan
    data[data==NoDataValue] = 0
    return data

def clean_train_set(data, file_name):
    cleaned_data = clean_data(data, 99.)
    scipy.io.savemat('./data/'+file_name, {'data':cleaned_data})
    print '+ saved train set.'

def clean_test_set(data, file_name):
    cleaned_missing = clean_data(data, 99.)
    cleaned_test = clean_data(cleaned_missing, 55.)
    cleaned_test.T
    scipy.io.savemat('./data/'+file_name, {'data':cleaned_test})
    print '+ saved test set.'

def clean_data_sets():
    clean_train_set(train_set, 'train_set.mat')
    print '+ cleaned training set...'
    clean_test_set(test_set, 'test_set.mat')
    print '+ cleaned test set...'


def pears(train_item, test_item):
    """Adapted from the _Collective Intelligence_ book p.13"""
    si ={}

    for i, val in enumerate(train_item):
        if val != 0:
            if test_item[i] != 0:
                si[i]=1

    n = len(si)

    if n==0: return 0

    sum1 = sum( [train_item[it] for it in si] )
    sum2 = sum( [test_item[it] for it in si] )

    sum1Sq = sum( [pow(train_item[it],2) for it in si] )
    sum2Sq = sum( [pow(test_item[it],2) for it in si] )

    pSum = sum([train_item[it]*test_item[it] for it in si])

    num = pSum - (sum1 * sum2/n)
    den = sqrt((sum1Sq - pow(sum1,2)/n) * (sum2Sq - pow(sum2, 2)/n))

    if den == 0: return 0

    r = num / den

    return r
    #print 'r: ', r
    
    

clean_data_sets()

train_mat = scipy.io.loadmat('./data/train_set.mat')
train_data = train_mat['data']
print '+ loaded cleaned training data.'
print train_data[0]

test_mat = scipy.io.loadmat('./data/test_set.mat')
test_data = test_mat['data']
print '+ loaded cleaned test data.'
print test_data[0]


print len(train_data)

# NOTE ---- using this method results in +.10 in sim score
pear_list = []
m = len(train_data)
for item in xrange(1):
    pear = np.corrcoef(train_data[item],test_data[0])
    data = (pear[0][1], item)
    pear_list.append(data)

print pear_list

# NOTE ---- using this method (hand-crafted pearson) results in -.10 in sim score
pears(train_data[0], test_data[0])

pear_data = []
for item in xrange(m):
    pear = pears(train_data[item], test_data[0])
    data = (pear,item)
    pear_data.append(data)

pear_data.sort()
pear_data.reverse()
print pear_data[:10]


#scipy.io.savemat('./data/pear_test.mat', {'data':pear_data})
