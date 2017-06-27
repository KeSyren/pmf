'''
Created on Nov 17, 2016

@author: qingwang
'''

'''
Created on Oct 23, 2016

@author: qingwang
'''
from mf.bayesian_matrix_factorization import example
import time
import math
import numpy as np
import numpy.random as rand
from numpy.linalg import inv, cholesky
from util.load_data import build_ml_1m, best_rating_user, build_rating_matrix, train_rating120


class EpsilonGreedy():
    
    def __init__(self, rating_matrix, user_feature, num_feature, item_features, lamda_p, epsilon, train):
        # rating rating_matrix 943 * 1682
        self.rating_matrix = rating_matrix 
        self.num_feature = num_feature
        # user_id item_id rating 100,000 * 3
        self.train = train
        
        # user feature
        self.user_feature = user_feature
        
        # latent variable
        self.mu_user = np.zeros((num_feature, 1), dtype='float64')
        
        # item features
        self.item_features = item_features
        
        # required parameters
        self.dic_userid_A_b = {}
        self.lamda_p = lamda_p
        # exploration-exploitation coefficient
        self.epsilon = epsilon
        
    def _estimate(self, train, tolerance=1e-5):
        # print train.shape
        # print self.item_features.shape
        sum_mse = 0
        iter = 0
        ret = []
        for id in xrange(len(train)):
            user_id = train[id][0] 
            max_rating = best_rating_user(self.rating_matrix, user_id)
            iter += 1
            print 'user_id ', user_id 
            
        # the algorithm will converge, but really slow
        # use MF's initialize latent parameter will be better
            if self.dic_userid_A_b.has_key(user_id):
                A = self.dic_userid_A_b.get(user_id)[0]
                b = self.dic_userid_A_b.get(user_id)[1]
            else:
                A = np.eye(self.num_feature, dtype='float64') * self.lamda_p
                b = np.zeros((self.num_feature, 1), dtype='float64')
            
            mu_user_feature = np.dot(inv(A), b)
            mu_user_feature_T = np.reshape(mu_user_feature, (self.num_feature,))
            
            # init item feature
            self.best_item_feature = np.zeros((self.num_feature, 1), dtype='float64')
            max_estimated_reward = - 100
            best_item  = 0
            
            threshold = rand.uniform(0,1)
            if threshold < self.epsilon:
                # random select an arm
                rand_id = rand.randint(0,len(self.item_features))
                best_item = rand_id
                w_item_feature_T = self.item_features[rand_id]
                w_item_feature = np.reshape(w_item_feature_T, (self.num_feature, 1))
                self.best_item_feature = w_item_feature
                max_estimated_reward = np.dot(mu_user_feature_T, w_item_feature)
            else:
                for item_id in xrange(len(self.item_features)):
                    # sample q_i from N(q_i|mu_i, Sigma_i)
                    w_item_feature_T = self.item_features[item_id]
                    w_item_feature = np.reshape(w_item_feature_T, (self.num_feature, 1))
                    
                    # estimated_reward = np.dot(sample_user_feature_T, sample_item_feature)
                    estimated_reward = np.dot(mu_user_feature_T, w_item_feature)
                    if estimated_reward >= max_estimated_reward:
                        best_item = item_id
                        self.best_item_feature = w_item_feature
                        max_estimated_reward = estimated_reward
                
            if max_estimated_reward > 5:
                max_estimated_reward = 5
            if max_estimated_reward < 0:
                max_estimated_reward = 0
            
            # reward
            real_reward = self.rating_matrix[user_id][best_item]
            print best_item
            if real_reward != 0:
                print best_item, real_reward, max_estimated_reward
            #sum_mse += math.pow((real_reward-max_estimated_reward), 2)
            sum_mse += math.pow((real_reward-max_rating), 2)
            # update A, b parameters
            A, b = self._update_user_hyperparams(A, b, real_reward, self.best_item_feature)
            self.dic_userid_A_b[user_id] = (A, b)       
            
            # compute RMSE
            print "iterations: %3d, train RMSE: %.6f" % (iter, sum_mse) 
            ret.append(str(iter) + ',' + str(sum_mse))
        return np.array(ret) 
             
    def _update_user_hyperparams(self, A, b, reward, best_item_feature):
        # _get_distribute_params mean and variance
        best_item_feature_T = np.reshape(best_item_feature, (self.num_feature,))
        A = A + np.outer(best_item_feature, best_item_feature_T)
        b = b + reward * best_item_feature
        return A, b
    
def test(num_feature, lamda_p, epsilon):
    bmf_model = example()
    rating_matrix = bmf_model.matrix
    user_feature = bmf_model.user_features
    print 'user_feature.shape', len(user_feature)
    # required parameters
    beta_item = bmf_model.beta_item
    N = bmf_model.num_item
    # item features from MAP solution
    item_features = bmf_model.item_features
    
    # split data to training & validation
    num_user, num_item, ratings = build_ml_1m()
    matrix = build_rating_matrix(num_user, num_item, ratings)
    train_matrix = train_rating120(matrix)  
    test = train_matrix[0]
    ratings = []
    for item in xrange(len(test)):
         rating = [0 for x in xrange(3)]
         if test[item] != 0:
             rating[1] = item
             rating[2] = test[item]
             rating = np.array(rating)
             ratings.append(rating)
    ratings = np.array(ratings)
   
    epsilon_model = EpsilonGreedy(rating_matrix, user_feature, num_feature, item_features, lamda_p, epsilon, ratings) 
    result = epsilon_model._estimate(ratings)
    epsilon_greedy_file = open('../result/ret_epsilon_greedy_test' + str(epsilon), 'w')
    for item in result:
        print>>epsilon_greedy_file, item
    epsilon_greedy_file.close()    

if __name__ == "__main__":
    test(30, 2, 0.1)
    test(30, 2, 0.05)