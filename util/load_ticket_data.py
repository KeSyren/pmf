"""
load data set

"""
import numpy as np
import scipy.sparse as sparse
import re


def build_ticket():
    """
    build movie lens 1M ratings from original ml_1m rating file.
    need to download and put ml_1m data in /data folder first.
    Source: http://www.grouplens.org/
    """
  
    print '\n1.loadind ticket data'
    with open("../data/ticket/ratings.data", "rb") as f:
        iter_lines = iter(f)
        ratings = []
        for line_num, line in enumerate(iter_lines):
            # format (user_id, item_id, rating)
            line = line.split('\t')[:3]
            line = [int(float(l)) for l in line]
            ratings.append(line)

    print "TOTAL #ratings :", line_num

    ratings = np.array(ratings)

    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] = ratings[:, (0, 1)] - 1
    num_user = max(ratings[:, 0]) + 1
    num_item = max(ratings[:, 1]) + 1
    
    print "MAX #user :", num_user
    print "MAX #item :", num_item
    return num_user, num_item, ratings


# def load_ml_1m():
#     """load Movie Lens 1M ratings from saved gzip file"""
#     import gzip
#     import cPickle
# 
#     file_path = 'data/ratings.gz'
#     with gzip.open(file_path, 'rb') as f:
#         print "load ratings from: %s" % file_path
#         num_user = cPickle.load(f)
#         num_item = cPickle.load(f)
#         ratings = cPickle.load(f)
# 
#         return num_user, num_item, ratings


def build_ticket_rating_matrix(num_user, num_item, ratings):
    """
    build dense ratings matrix from original ticket rating file.
    """

    print '\n2.build matrix'
    # sparse matrix
    # matrix = sparse.lil_matrix((num_user, num_item))
    # dense matrix
    matrix = np.zeros((num_user, num_item), dtype='int8')
    
    for item_id in xrange(num_item):
        data = ratings[ratings[:, 1] == item_id]
        if data.shape[0] > 0:
            matrix[data[:, 0], item_id] = data[:, 2]
            
    print "RATING_MATRIX #row :", len(matrix)
    print "RATING_MATRIX #column: ", len(matrix[0])
    return matrix


def build_sparse_matrix(num_user, num_item, ratings):
    # TODO: have not tested it yet. will test after algorithm support sparse
    # matrix
    print '\nbuild sparse matrix'
    matrix = sparse.lil_matrix((num_user, num_item))
    for item_id in xrange(num_item):
        data = ratings[ratings[:, 1] == item_id]
        if data.shape[0] > 0:
            # for sparse matrix
            matrix[data[:, 0], item_id] = np.array([data[:, 2]]).T
  
    return matrix

def best_rating_user(rating_matrix, user_id):
    rating_user = rating_matrix[user_id, :]
    max_rating = -1
    for rating in rating_user:
        if rating > max_rating:
            max_rating = rating
    return max_rating  


def train_rating120(rating_matrix):
    ret = []
    count = 0
    for user_id in xrange(len(rating_matrix)):
        for item_id in xrange(len(rating_matrix[user_id,:])):
            if rating_matrix[user_id][item_id] != 0:
                count = count + 1
        if count >= 1:
            ret.append(rating_matrix[user_id,:])
        count = 0
    ret = np.array(ret)
    return ret

 
def main():
    num_user, num_item, ratings = build_ticket()
    matrix = build_ticket_rating_matrix(num_user, num_item, ratings)
    
if __name__ == "__main__":
    main()