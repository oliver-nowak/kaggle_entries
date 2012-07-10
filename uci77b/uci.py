import scipy.io
import scipy.spatial.distance as spsd
import numpy as np
import scipy.stats
import csv

mat = scipy.io.loadmat('./data/kaggle77b_trainset.mat')

test_mat = scipy.io.loadmat('./data/kaggle77b_testset.mat')

#row1 = mat['trainset'][0]
#row2 = mat['trainset'][1]
#row_set = mat['trainset'][:10]


#test_row1 = test_mat['testset'][0]
#test_row2 = test_mat['testset'][1]


#test_array = np.array([row1, test_row1])

train_set = mat['trainset']
test_set = test_mat['testset']
#test_row_set = test_mat['testset'][:2]


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

def compile_pear_scores():
    # NOTE VALIDATED WITH EXCEL
    pear_data = np.zeros( (3000, 21983) )
    m = len(train_data)
    k = len(test_data)
    row_count = 0
    col_count = 0
    for test_subject in xrange(k):
        for item in xrange(m):
            pear = np.corrcoef(train_data[item],test_data[test_subject])
            pear_data[row_count][col_count] = pear[0][1]
            col_count += 1
        row_count += 1
        print 'cols : %s' % col_count
        print '...calculated %s of 3000 rows' % row_count
        col_count = 0

    print 'preparing to save pear matrix...'
#    scipy.io.savemat('./data/pear_set.mat', {'data': pear_data})
    scipy.io.savemat('./data/y1_y2_pear_set.mat', {'data': pear_data})
    print '...pear matrix saved.'
    print pear_data[0][0]

    print 'Exiting.'
    exit()

def compile_item_pear_scores():
    #TODO: this should be validated in excel to confirm the right coefficients
    train_items = train_data.T
    print '+ train_items shape : %s' % str( train_items.shape)

    pear_data = np.corrcoef( train_items )
    
    print '+ item pear data shape : %s ' % str(pear_data.shape)
    print '+ preparing to save item pear matrix...'

    scipy.io.savemat('./data/item_pear_set.mat', {'data': pear_data})
    print '+ saved item pear matrix.'
    print pear_data[0][:2]

    print 'Exiting.'
    exit()


def add_y1_pear_scores():
    """This function adds the predicted y1 scores back into the test-set as a 
    'feed-forward' function to get better y2 and y3 predictions. (EXPERIMENTAL)"""
    
    # load predictions
    y_mat = scipy.io.loadmat('./data/predictions.mat')
    y_data = y_mat['data'] 
    # iterate through test-set
    for i,row in enumerate(test_data):
        #print i
        #print row
        # find y1 in predictions data set
        y1 = y_data[i][0]
        y2 = y_data[i][1]

        # find y1 index in prediction indices data set
        y1_ind = pred_data[i][0]
        y2_ind = pred_data[i][1]

        # copy value of y1 into test set at the correct index
        row[y1_ind] = y1
        row[y2_ind] = y2

    # save new test_set
    scipy.io.savemat('./data/y1_test_set.mat', {'data': test_data})
    print '+ saved y1_test_set.'



def calculate_item_ratings():
    #get y indices
    #test_user = 0
    #y1x = pred_data[test_user][0]
    #y2x = pred_data[test_user][1]
    #y3x = pred_data[test_user][2]
    #print y1x
    
    #y1_pearx = item_pear_data[y1x][:]
    #y2_pearx = item_pear_data[y2x][:]
    #y3_pearx = item_pear_data[y3x][:]
    #print y1_pearx.shape
    
    #print test_data[0].shape

    k = test_data.shape[0]
    #print k

    item_scores = np.zeros( (3000, 3) )
    for user_row in xrange(k):
        y1 = pred_data[user_row][0]
        y2 = pred_data[user_row][1]
        y3 = pred_data[user_row][2]

        y1_pear = item_pear_data[y1][:]
        y2_pear = item_pear_data[y2][:]
        y3_pear = item_pear_data[y3][:]
        
        y1_rating = np.dot( test_data[user_row], y1_pear )
        y2_rating = np.dot( test_data[user_row], y2_pear )
        y3_rating = np.dot( test_data[user_row], y3_pear )

        y1_norm = np.sum(y1_pear)
        y2_norm = np.sum(y2_pear)
        y3_norm = np.sum(y3_pear)

        y1_norm_score = y1_rating / y1_norm
        y2_norm_score = y2_rating / y2_norm
        y3_norm_score = y3_rating / y3_norm
        
        item_scores[user_row][0] = y1_norm_score
        item_scores[user_row][1] = y2_norm_score
        item_scores[user_row][2] = y3_norm_score
        

    #test = np.dot( test_data[0], y1_pearx )
    #print test

    #norm = np.sum(y1_pearx)
    #print 'norm value: %s' % norm

    #norm_score = test / norm
    #print 'norm_score : %s ' % norm_score

    print 'item_score : %s ' % item_scores[0]

    scipy.io.savemat('./data/item_predictions.mat', {'data': item_scores})
    print '+ saved item_predictions.'

    csv_writer = csv.writer(open('./data/item_predictions.csv', 'wb'), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for row in item_scores:
        csv_writer.writerow(row)
    print '+ saving predictions csv file.'


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
print '+ y1 should be 0: %s ' % test_data[0][7]
print '+ y2 should be 0: %s ' % test_data[0][34]


pred_mat = scipy.io.loadmat('./data/prediction_indices.mat')
pred_data = pred_mat['data']
print '+ loaded prediction indices.'
print '+ total prediction indices : %s ' % len(pred_data)
print pred_data[0]

pear_mat = scipy.io.loadmat('./data/pear_set.mat')
pear_data = pear_mat['data']
print '+ loaded pear data.'
print '+ total pear scores : %s ' % len(pear_data)
print pear_data[0]

item_pear_mat = scipy.io.loadmat('./data/item_pear_set.mat')
item_pear_data = item_pear_mat['data']
print '+ loaded item pear data.'
print '+ total pear scores : %s' % str(item_pear_data.shape)
print item_pear_data[0][:2]

#add_y1_pear_scores()

#load recompiled test_data that includes the y1 column
#test_mat = scipy.io.loadmat('./data/y1_test_set.mat')
#test_data = test_mat['data']
#print '+ loaded recompiled y1_test_set.'
#print '+ should be 0.9060 : %s' % test_data[0][7]
#print '+ should be 0.2411 : %s' % test_data[0][34]


#compile_pear_scores()

#compile_item_pear_scores()

calculate_item_ratings()

exit()





##
#   USER_BASED COLLABORATIVE FILTER CODE 
##

# initialize some start vars
row = 0
m   = len(train_data)
k   = len(test_data)


# first missing joke of _first test data row_ to predict
y1 = pred_data[row][0]
y2 = pred_data[row][1]
y3 = pred_data[row][2]

print '+ current row : %s' % row
print '+ indice to predict: %s' % y1

# create a vector containing all y1 ratings in training set
# each row is one rating label from the training set
y1_vec = np.zeros( (m,1) )
y2_vec = np.zeros( (m,1) )
y3_vec = np.zeros( (m,1) )
ind = 0

print '+ total m to init : %s ' % m
print '+ total train_data : %s ' % len(train_data)
for item in train_data:
    y1_rating = item[y1]
    y1_vec[ind] = y1_rating

    y2_rating = item[y2]
    y2_vec[ind] = y2_rating

    y3_rating = item[y3]
    y3_vec[ind] = y3_rating

    ind += 1

# calculate the normalize scaler by summing every rating of yi
y1_norm_scalar = np.sum( y1_vec )
print 'y1_norm_scalar : %s ' % y1_norm_scalar
y2_norm_scalar = np.sum( y2_vec )
print 'y2_norm_scalar : %s ' % y2_norm_scalar
y3_norm_scalar = np.sum( y3_vec )
print 'y3_norm_scalar : %s ' % y3_norm_scalar

# for every test row, calculate the dot product of the row and the ratings vector
# this will also sum the values, leaving one unnormalized scalar value as the prediction
# for that row (test_subject) of yx.
# this prediction is normalized using the normalize_scalar calculated above in order
# to get the correct prediction value of yx.

y_list = np.zeros( (k, 3) )
pear_ind = 0
for row in pear_data:
    # (1 x 21983).(21983 x 1)
    # results in one unnormalized number per test_set item (~3000 rows, 1 column) 
    # normalized by the norm_yx scalar for yx rating

    y1_sim = np.dot(row, y1_vec) 
    norm_y1 = y1_sim / y1_norm_scalar
    y_list[pear_ind][0] = norm_y1
    
    y2_sim = np.dot(row, y2_vec)
    norm_y2 = y2_sim / y2_norm_scalar
    y_list[pear_ind][1] = norm_y2
    

    y3_sim = np.dot(row, y3_vec)
    norm_y3 = y3_sim / y3_norm_scalar
    y_list[pear_ind][2] = norm_y3
    
    pear_ind += 1

scipy.io.savemat('./data/y2_predictions.mat', {'data': y_list})
print '+ saving predictions matrix.'

csv_writer = csv.writer(open('./data/y2_predictions.csv', 'wb'), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

for row in y_list:
    csv_writer.writerow(row)
print '+ saving predictions csv file.'

#TASKS
######
#DONE: validate pearson calculation in excel - validated in excel
#DONE: get erased prediction indices from test_set
#DONE: automate pearson calculation for test_set data
#DONE: add remaining code to calculate recommendations
#TODO: see if we can increase the float precision by configuring the numpy dtype
#DONE: validate that entire calculated float precision is being written to csv
#DONE: validate that EXCEL is _not_ rounding float values - values look rounded in EXCEL but not in console when mat is loaded
#DONE: test if iterative calculation on remaining missing prediction indices are affected
#TODO: implement item-based filtering to see if prediction accuracy is increased
#TODO: add test-set to item-based pearson calculation to see if prediction accuracy is increased

