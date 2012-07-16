import scipy.io
import scipy.spatial.distance as spsd
import numpy as np
import scipy.stats
import csv
import math

mat = scipy.io.loadmat('./data/kaggle77b_trainset.mat')

add_mat = scipy.io.loadmat('./data/add_train_data.mat')

test_mat = scipy.io.loadmat('./data/kaggle77b_testset.mat')

#row1 = mat['trainset'][0]
#row2 = mat['trainset'][1]
#row_set = mat['trainset'][:10]


#test_row1 = test_mat['testset'][0]
#test_row2 = test_mat['testset'][1]


#test_array = np.array([row1, test_row1])

train_set = np.array(mat['trainset'], dtype=np.float64)
add_train_set = np.array(add_mat['data'], dtype=np.float64)
test_set = np.array(test_mat['testset'], dtype=np.float64)
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
 
def weighted_std(arr, weights):
    N = arr.shape[0]
    #print 'N : %s ' % N

    wtot = weights.sum()
    
    wmean = (weights * arr).sum()/wtot

    #werr2 = ( weights**2 * (arr-wmean)**2).sum()
    #werr = np.sqrt(werr2)/wtot

    #error imp 2
    #werr = 1.0 / np.sqrt(wtot)
    
    wvar = ( weights *(arr-wmean)**2).sum()/( ((N - 1) * wtot)/N )
    #wsdev = np.sqrt(wvar)
    wsdev = wvar
    return wsdev
    
#    average = np.average(values, weights=weights)
#    variance = np.dot(weights, (values - average)**2) / weights.sum()
#    return math.sqrt(variance)

def clean_data(data, NoDataValue):
    """Replace the NoDataValue with np.nan in order to avoid data skew when we 
    calculate the Pearson correlation"""
    data[data==NoDataValue] = 0
    return data

def clean_train_set(data, file_name):
    print 'shape of train set : %s' % str(data.shape)
    print 'shape of add train set : %s' % str(add_train_set.shape)
    cleaned_data = clean_data(data, 99.)
    #cleaned_add_train_data = clean_data(add_train_set, 99.)

    #appended_arr = np.append(cleaned_data, cleaned_add_train_data, axis=0)

    #print 'shape of appended_arr : %s' % str(appended_arr.shape)
    #scipy.io.savemat('./data/'+file_name, {'data':appended_arr})
    scipy.io.savemat('./data/'+file_name, {'data': cleaned_data})
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

def compile_user_pear_scores():
    """Build Pearson similiarity matrix based on user comparision"""
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
    """Build Pearson similiarity matrix based on item comparision"""
    train_items = train_data.T
    print '+ train_items shape : %s' % str( train_items.shape)
    
    #test_items = test_data.T
    #print '+ test_items shape : %s' % str( test_items.shape )

    # appending test items to train item set - EXPERIMENTAL
    #train_items = np.append(train_items, test_items, axis=1)
    #print '+ append train_items : %s' % str( train_items.shape )

    pear_data = np.corrcoef( train_items, bias=1 )
    
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


def compile_avg_ratings():
    """Compiles mean ratings of each user."""
    k = train_data.shape[0]
    
    # remove 0-rated elements so they dont affect the mean calculation
    # calculate the mean rating across users
    avg_rating = []
    for i in xrange(k):
        distilled_user = [x for x in train_data[i] if x != 0.]
        avg_rating.append( np.mean(distilled_user, dtype=np.float64) )
    
    return avg_rating

def compile_avg_item_ratings():
    """Compiles mean ratings of each item."""
    k = train_data.shape[1]
    
    avg_item_rating = []
    for i in xrange(k):
        distilled_item = [x for x in train_data[:][i] if x != 0.]
        avg_item_rating.append( np.mean(distilled_item, dtype=np.float64) )

    return avg_item_rating


def compile_adj_cosine_sim():
    """Build Adjusted Cosine similiarity matrix based on item comparisions."""
    avg_rating = compile_avg_ratings()
    print 'avg_rating size: %s' % len( avg_rating )
    print 'avg_rating[:10] : %s' % avg_rating[:10]
    
    train_items = train_data.T
    k = train_items.shape[0]
    m = train_items.shape[1]
    print 'k : %s' % k
    print 'm : %s' % m

    sim_data = np.zeros( (100, 100), dtype=np.float64 )
    #k = 2

    # iterate through matrix of 100 items and calculate the sim score
    for i in xrange(k):
        for j in xrange(k):
            # subtract the avg user rating element-wise 
            item_i = train_items[i] - avg_rating
            item_j = train_items[j] - avg_rating

            # find the dot product between the items
            similiarity_score = np.dot(item_i, item_j)
            #print 'sim : %s' % similiarity_score

            # find the dot product of the squared difference
            sq_diff_term_i = np.dot(item_i, item_i)
            sq_diff_term_j = np.dot(item_j, item_j)

            # reduce down via square-root
            diff_term_i = math.sqrt(sq_diff_term_i)
            diff_term_j = math.sqrt(sq_diff_term_j)

            # multiply the terms in order to find the normalizer
            total_diff = diff_term_i * diff_term_j

            # calculate the adjusted cosine score
            norm_sim_score = similiarity_score / total_diff
            #print 'norm_sim_score : %s' % norm_sim_score
            
            # save the score 
            sim_data[i][j] = norm_sim_score

    #print sim_data
    scipy.io.savemat('./data/adj_cosine_sim.mat', {'data':sim_data})
    print '+ saved adjusted cosine similiarity matrix at single-precision.'

    return sim_data

def calculate_item_ratings_w_adj_cosine():
    k = test_data.shape[0]

    #sim_scores = compile_adj_cosine_sim()
    
    sim_scores = item_cos_data
    item_scores = np.zeros( (3000, 3), dtype=np.float64 )

    #k = 2

    for user_row in xrange(k):
        
        # find prediction indices for a particular user
        y1 = pred_data[user_row][0]
        y2 = pred_data[user_row][1]
        y3 = pred_data[user_row][2]
        
        # find the adjusted cosine similiarity at yx
        y1_cos = sim_scores[y1]
        y1_cos[y1] = 0.

        y2_cos = sim_scores[y2]
        y2_cos[y2] = 0.

        y3_cos = sim_scores[y3]
        y3_cos[y2] = 0.

        y1_rating = np.dot( test_data[user_row], y1_cos )
        y2_rating = np.dot( test_data[user_row], y2_cos )
        y3_rating = np.dot( test_data[user_row], y3_cos )

        y1_norm = np.sum(np.absolute(y1_cos)) - 1.
        y2_norm = np.sum(np.absolute(y2_cos)) - 1.
        y3_norm = np.sum(np.absolute(y3_cos)) - 1.

        y1_norm_score = y1_rating / y1_norm
        y2_norm_score = y2_rating / y2_norm
        y3_norm_score = y3_rating / y3_norm

        item_scores[user_row][0] = y1_norm_score
        item_scores[user_row][1] = y2_norm_score
        item_scores[user_row][2] = y3_norm_score


    scipy.io.savemat('./data/item_predictions.mat', {'data': item_scores})
    print '+ saved item_predictions.'

    csv_writer = csv.writer(open('./data/adj_cos_item_predictions.csv', 'wb'), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for row in item_scores:
        csv_writer.writerow(row)
    print '+ saving predictions csv file.'

        

def calculate_item_ratings_w_pearson():
    k = test_data.shape[0]
    print 'test_data.shape : %s' % str(test_data.shape)
    item_scores = np.zeros( (3000, 3), dtype=np.float64 )
    for user_row in xrange(k):
        y1 = pred_data[user_row][0]
        y2 = pred_data[user_row][1]
        y3 = pred_data[user_row][2]

        y1_pear = item_pear_data[y1][:]
        y1_pear[y1] = 0.
        
        y2_pear = item_pear_data[y2][:]
        y2_pear[y2] = 0.
        
        y3_pear = item_pear_data[y3][:]
        y3_pear[y3] = 0.

        y1_rating = 0.
        y2_rating = 0.
        y3_rating = 0.     

        

        y1_rating = np.dot( test_data[user_row], y1_pear )
        y2_rating = np.dot( test_data[user_row], y2_pear )
        y3_rating = np.dot( test_data[user_row], y3_pear )


        y1_norm = np.sum(np.absolute(y1_pear)) - 1.
        y2_norm = np.sum(np.absolute(y2_pear)) - 1.
        y3_norm = np.sum(np.absolute(y3_pear)) - 1.
            
        y1_norm_score = y1_rating / y1_norm
        y2_norm_score = y2_rating / y2_norm
        y3_norm_score = y3_rating / y3_norm

        item_scores[user_row][0] = y1_norm_score
        item_scores[user_row][1] = y2_norm_score
        item_scores[user_row][2] = y3_norm_score
        
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
print train_data[-1]

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

#pear_mat = scipy.io.loadmat('./data/pear_set.mat')
#pear_data = pear_mat['data']
#print '+ loaded pear data.'
#print '+ total pear scores : %s ' % len(pear_data)
#print pear_data[0]

#compile_item_pear_scores()

item_pear_mat = scipy.io.loadmat('./data/item_pear_set.mat')
item_pear_data = item_pear_mat['data']
print '+ loaded item pear data.'
print '+ total pear scores : %s' % str(item_pear_data.shape)
print item_pear_data[0][:]

#item_cos_mat = scipy.io.loadmat('./data/adj_cosine_sim.mat')
#item_cos_data = item_cos_mat['data']
#print '+ loaded item cos data.'
#print '+ total item cos scores : %s' % str(item_cos_data.shape)
#print item_cos_data[0][:]

#add_y1_pear_scores()

#load recompiled test_data that includes the y1 column
#test_mat = scipy.io.loadmat('./data/y1_test_set.mat')
#test_data = test_mat['data']
#print '+ loaded recompiled y1_test_set.'
#print '+ should be 0.9060 : %s' % test_data[0][7]
#print '+ should be 0.2411 : %s' % test_data[0][34]


#compile_pear_scores()

#compile_item_pear_scores()

#test_y = np.array([2,3,5,7,11,13,17,19,23])
#test_w = np.array([1,1,0,0,4,1,2,1,0])
#print weighted_std(test_y, test_w)

compile_avg_item_ratings()
#calculate_item_ratings_w_pearson()
#compile_adj_cosine_sim()
#calculate_item_ratings_w_adj_cosine()
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
#DONE: see if we can increase the float precision by configuring the numpy dtype
#DONE: validate that entire calculated float precision is being written to csv
#DONE: validate that EXCEL is _not_ rounding float values - values look rounded in EXCEL but not in console when mat is loaded
#DONE: test if iterative calculation on remaining missing prediction indices are affected
#DONE: implement item-based filtering to see if prediction accuracy is increased
#DONE: add test-set to item-based pearson calculation to see if prediction accuracy is increased
#DONE: try weighted std instead of weighted sum for normalizing predictions
#TODO: try capping neighborhood to <=30 when calculating Pearson
