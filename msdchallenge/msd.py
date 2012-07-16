import scipy.io
import scipy.sparse
import scipy.stats
import numpy as np
import math
import csv

def compile_item_pear_scores(data):
    print '+ compiling item pear scores for data: %s' % str(data.shape)

    pear_data = np.corrcoef( data, bias=1 )

    print '+ item pear data shape : %s' % str(pear_data.shape)
    print '+ preparing to save item pear matrix...'

    scipy.io.savemat('./data/item_pear_set.mat', {'data': pear_data})
    print '+ saved item pear matrix.'
    print pear_data[0][:2]

    print 'Exiting.'
    exit()


def normalize_song_counts():
    """Song counts are a proxy for _ratings_. This method will normalize
    song counts across the test set in order to get a consistent rating
    unit."""

    csv_reader = csv.reader(open('./data/user_song_count.txt', 'rb'), delimiter=',', quotechar='|')
    
    total_count = 0
    total_rows = 0
    
    train_mat = scipy.sparse.lil_matrix((110000, 386213))

    for row in csv_reader:
        user, song, count = row
        total_count += int(count)
        total_rows += 1
        
        user_idx = int(user)
        song_idx = int(song)
        rating   = int(count)
#        print user_idx
#        print song_idx
#        print rating
        # user and song are 1-indexed; but array is 0-indexed
        train_mat[user_idx-1,song_idx-1] = rating
    
    scipy.io.mmwrite(file('./data/train_mat.txt', 'wb'), train_mat)
    #print '+ train_mat [0,0] : ', train_mat[0][0]
    print '+ train_set shape : %s' % str(train_mat.shape) 
    print '+ total count : %s' % total_count
    print '+ total rows : %s' % total_rows

    norm_value = total_count / total_rows

    print '+ total norm_value %s' % norm_value
    
    # calculate Pearson (fast)
    #pear_data = np.corrcoef(train_mat)
    #rows = 110000
    #ms = train_mat.mean(axis=1)[(slice(None,None,None),None)]
    #datam = train_mat - ms
    #datass = np.sqrt(scipy.stats.ss(datam,axis=1))
    #for i in xrange(rows):
    #    temp = np.dot(datam[i:],datam[i].T)
    #    rs = temp / (datass[i:]*datass[i])



    return norm_value

rating_unit = normalize_song_counts()

