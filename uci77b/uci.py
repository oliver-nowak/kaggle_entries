import scipy.io
import scipy.spatial.distance as spsd
import numpy as np
import scipy.stats

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
    data[data==NoDataValue] = 0
    return data

def clean_train_set(data, file_name):
    cleaned_data = clean_data(data, 99.)
    scipy.io.savemat('./data/'+file_name, {'data':cleaned_data})
    print '+ cleaned and saved train set.'

def clean_test_set(data, file_name):
    cleaned_missing = clean_data(data, 99.)
    cleaned_test = clean_data(cleaned_missing, 55.)
    cleaned_test.T
    scipy.io.savemat('./data/'+file_name, {'data':cleaned_test})
    print '+ cleaned and saved test set.'

def clean_data_sets():
    clean_train_set(train_set, 'train_set.mat')
    clean_test_set(test_set, 'test_set.mat')

def export_to_file(data, file_name):
    print '+ exporting %s' % file_name
    with open('./data/'+file_name, 'w') as f:
        for value in data:
            #print value
            f.write(str(value))
            f.write(',')
    
    print '+ %s saved.' % file_name

def get_prediction_indices():
    """This gets the prediction indices marked as '55' in the test set that are
    required for the kaggle entry and saves them to a separate lookup table."""
    indices = []
    test_mat = scipy.io.loadmat('./data/kaggle77b_testset.mat')
    test_set = test_mat['testset']
    for row in test_set:
        predict = []
        for i,item in enumerate(row):
            if item == 55.:
                predict.append(i)

        t = (predict)
        indices.append(t)

    scipy.io.savemat('./data/prediction_indices.mat', {'data':indices})
    print '+ saved prediction_indices.mat'

get_prediction_indices()

clean_data_sets()

train_mat = scipy.io.loadmat('./data/train_set.mat')
train_data = train_mat['data']
print '+ loaded cleaned training data.'
print '+ total train_data records : %s' % len(train_data)
print train_data[0]

test_mat = scipy.io.loadmat('./data/test_set.mat')
test_data = test_mat['data']
print '+ loaded cleaned test data.'
print '+ total test_data records : %s ' % len(test_data)
print test_data[0]


pred_mat = scipy.io.loadmat('./data/prediction_indices.mat')
pred_data = pred_mat['data']
print '+ loaded prediction indices.'
print '+ total prediction indices : %s ' % len(pred_data)
print pred_data[0]

# exporting first row of test and train data to validate pearson calculation in excel
#export_to_file(train_data[0], 'train_data_row1.csv')
#export_to_file(test_data[0], 'test_data_row1.csv')


# NOTE ---- using this method results in +.10 in sim score
# NOTE VALIDATED WITH EXCEL
pear_list = []
m = len(train_data)
for item in xrange(m):
    pear = np.corrcoef(train_data[item],test_data[0])
    data = (pear[0][1], item)
    pear_list.append(data)

print pear_list[0]


#TASKS
######
#DONE: validate pearson calculation in excel - validated in excel
#DONE: get erased prediction indices from test_set
#TODO: automate pearson calculation for test_set data
#TODO: add remaining code to calculate recommendations
#TODO: test if iterative calculation on remaining missing prediction indices are affected
#TODO: implement item-based filtering to see if prediction accuracy is increased

