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


class ThompsonSampling():
    
    def __init__(self, rating_matrix, user_feature, num_feature, item_features, delta, lamda_p, train):
        # rating rating_matrix 943 * 1682
        self.rating_matrix = rating_matrix 
        self.num_feature = num_feature
        # user_id item_id rating 100,000 * 3
        self.train = train
        
        # user feature
        self.user_feature = user_feature
        
        # latent variable
        self.mu_user = np.zeros((num_feature, 1), dtype='float64')
        self.Sigma_user = np.eye(num_feature, dtype='float64')
        
        # tuple(mu_i, Sigma_i)
        self.item_features = item_features
        
        # required parameters
        self.dic_userid_A_b = {}
        
        self.delta = delta
        self.lamda_p = lamda_p
        
               
    def _estimate(self, train, tolerance=1e-5):
        print train.shape
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
            
            # update user parameter
            self.mu_user, self.Sigma_user = self._get_distribute_params(A, b, self.delta)
            # sample p
            lam = cholesky(self.Sigma_user)
            lam = lam.T
            
            sample_user_feature = self.mu_user + np.dot(lam, rand.randn(self.num_feature, 1))
            sample_user_feature_T = np.reshape(sample_user_feature, (self.num_feature,))
            
            # sample q*
            self.best_item_feature = np.zeros((self.num_feature, 1), dtype='float64')
            max_estimated_reward = -100
            best_item  = 0
            
            for item_id in xrange(len(self.item_features)):
                temp_mu_Sigma = self.item_features[item_id]
                mu_item = temp_mu_Sigma[0]
                Sigma_item = temp_mu_Sigma[1]
                lam = cholesky(Sigma_item)
                lam = lam.T
                # sample q_i from N(q_i|mu_i, Sigma_i)
                sample_item_feature = mu_item + np.dot(lam, rand.randn(self.num_feature, 1))
                sample_item_feature_T = np.reshape(sample_item_feature, (self.num_feature,))
                
                estimated_reward = np.dot(sample_user_feature_T, sample_item_feature)
                
                if estimated_reward >= max_estimated_reward:
                    best_item = item_id
                    self.best_item_feature = sample_item_feature
                    max_estimated_reward = estimated_reward
            
            if max_estimated_reward > 5:
                max_estimated_reward = 5
            if max_estimated_reward < 0:
                max_estimated_reward = 0
            # reward
            real_reward = self.rating_matrix[user_id][best_item]
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
             
    
    def _get_distribute_params(self, A, b, delta):
        # _get_distribute_params mean and variance
        mu_user = np.dot(inv(A), b)
        Sigma_user = inv(A)*math.pow(delta, 2)
        return mu_user, Sigma_user
    
    def _update_user_hyperparams(self, A, b, reward, best_item_feature):
        # _get_distribute_params mean and variance
        best_item_feature_T = np.reshape(best_item_feature, (self.num_feature,))
        A = A + np.outer(best_item_feature, best_item_feature_T)
        b = b + reward * best_item_feature
        return A, b
    
def test(num_feature=30, delta=1, lamda_p=2):
    bmf_model = example()
    rating_matrix = bmf_model.matrix
    user_feature = bmf_model.user_features
    print 'user_feature.shape', len(user_feature)
    # required parameters
    beta_item = bmf_model.beta_item
    N = bmf_model.num_item
    # item mean and variance
    mu_Sigma_items = bmf_model.ret_mu_Sigma_item
    
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
     
    ts_model = ThompsonSampling(rating_matrix, user_feature, num_feature, mu_Sigma_items, delta, lamda_p, ratings) 
    result = ts_model._estimate(ratings)
    thompson_sampling_file = open('../result/ret_thompson_sampling_test', 'w')
    for item in result:
        print>>thompson_sampling_file, item
    thompson_sampling_file.close()    

if __name__ == "__main__":
    test()