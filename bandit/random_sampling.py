'''
Created on Oct 24, 2016

@author: qingwang
'''
import time
import math
import numpy as np
import random
from util.load_data import build_ml_1m, build_rating_matrix, best_rating_user, train_rating120

class RandomSampling():
    def __init__(self, rating_matrix, train):
        # rating rating_matrix 943 * 1682
        self.rating_matrix = rating_matrix 
        # user_id item_id rating 100,000 * 3
        self.train = train
    
    def _estimate(self, train, num_item):
        sum_mse = 0
        iter = 0
        ret = []
        for id in xrange(len(train)):
            user_id = train[id][0] 
            max_rating = best_rating_user(self.rating_matrix, user_id)
            iter += 1
            print 'user_id ', user_id 
            
            best_item = self._random_select(num_item)
            print best_item
            # reward
            real_reward = self.rating_matrix[user_id][best_item]
#             if real_reward != 0:
#                 print real_reward
            #sum_mse += math.pow((real_reward-max_estimated_reward), 2)
            sum_mse += math.pow((real_reward-max_rating), 2)
            print "iterations: %3d, train RMSE: %.6f" % (iter, sum_mse) 
            
            ret.append(sum_mse)
        return ret
    
    def _random_select(self, num_item): 
        random_arm = random.randint(0, num_item-1)
        return random_arm
        #pass
        
def test():
    num_user, num_item, ratings = build_ml_1m()
    np.random.shuffle(ratings)
    matrix = build_rating_matrix(num_user, num_item, ratings)
    train_matrix = train_rating120(matrix)  
    test = train_matrix[0]
#     ratings = []
#     for item in xrange(len(test)):
#          rating = [0 for x in xrange(3)]
#          if test[item] != 0:
#              rating[1] = item
#              rating[2] = test[item]
#              rating = np.array(rating)
#              ratings.append(rating)
#     ratings = np.array(ratings)
     
    
    random_model = RandomSampling(matrix, ratings)
    result = random_model._estimate(ratings, num_item)
    
    random_sampling_file = open('../result/ret_random_sampling', 'w')
    for item in result:
        print>>random_sampling_file, item
    random_sampling_file.close()
    
if __name__ == "__main__":
    test()
    
    
    
    