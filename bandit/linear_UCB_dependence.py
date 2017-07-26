'''
Created on Oct 23, 2016

@author: qingwang
'''
from mf.bayesian_matrix_factorization import example
from sklearn.metrics.pairwise import cosine_similarity
import time
import math
import numpy as np
import pandas as pd
import numpy.random as rand
from numpy.linalg import inv, cholesky
from util.load_data import build_ml_len, best_rating_user, build_rating_matrix, train_rating120


class LinearUCBDependence():
    
    def __init__(self, rating_matrix, user_feature, num_feature, item_features, delta, lamda_p, alpha, train):
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
        
        # item features
        self.item_features = item_features
        
        # required parameters
        self.dic_userid_A_b = {}
        self.delta = delta
        self.lamda_p = lamda_p
        # exploration-exploitation coefficient
        self.alpha = alpha
        
        self.similarity_matrix = [[0 for i in range(len(self.item_features))] for j in range(len(self.item_features))]
        
    def _cal_similarity_matrix(self, item_features):
        for i in xrange(len(item_features)):
            print '%d row' %(i+1)
            self.similarity_matrix[i][i] = 1
            j = i + 1
            while j < len(item_features):
                similarity_ij = cosine_similarity(item_features[i], item_features[j])
                self.similarity_matrix[i][j] = similarity_ij[0][0]
                self.similarity_matrix[j][i] = similarity_ij[0][0]
                j = j + 1
        
        mat = np.matrix(self.similarity_matrix)
        dataframe = pd.DataFrame(data=mat.astype(float))
        dataframe.to_csv('../result/item_feature_similarity.csv', sep=',', header=False, float_format='%.2f', index=False)
    
    def _load_similarity_matrix(self):
        df = pd.read_csv('../result/item_feature_similarity.csv', sep=',', header = None)
        self.similarity_matrix = np.array(df)
          
    def _estimate(self, train, tolerance=1e-5):
#         self._cal_similarity_matrix(self.item_features)
        self._load_similarity_matrix()
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
            
            # update user parameter
            self.mu_user, self.Sigma_user = self._get_distribute_params(A, b, self.delta)
            # sample p
            lam = cholesky(self.Sigma_user)
            lam = lam.T
            
            w_user_feature = np.dot(inv(A), b)
            w_user_feature_T = np.reshape(w_user_feature, (self.num_feature,))
            
            # sample q*
            self.best_item_feature = np.zeros((self.num_feature, 1), dtype='float64')
            max_estimated_reward = -100
            best_item  = -1
            
            for item_id in xrange(len(self.item_features)):
               
                # sample q_i from N(q_i|mu_i, Sigma_i)
                w_item_feature_T = self.item_features[item_id]
                w_item_feature = np.reshape(w_item_feature_T, (self.num_feature, 1))
                
                # estimated_reward = np.dot(sample_user_feature_T, sample_item_feature)
                confidence = math.sqrt(np.dot(np.dot(w_item_feature_T, self.Sigma_user), w_item_feature))
                estimated_reward = np.dot(w_user_feature_T, w_item_feature) + self.alpha*confidence
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
            if real_reward != 0:
                print best_item, real_reward, max_estimated_reward
            #sum_mse += math.pow((real_reward-max_estimated_reward), 2)
            sum_mse += math.pow((real_reward-max_rating), 2)
            # update A, b parameters
            A, b = self._update_user_hyperparams(A, b, real_reward, self.best_item_feature)
            self.dic_userid_A_b[user_id] = (A, b)       
            
            # update by the similarity items
            similarity_items = self.similarity_matrix[best_item]
            top_k_similarity_idx = self._top_k_similarity(similarity_items, 11)
            for idx in top_k_similarity_idx:
                item_id = idx
                if item_id == best_item:
                    continue
                reward = self.rating_matrix[user_id][item_id]
                w_sitem_feature_T = self.item_features[item_id]
                w_sitem_feature = np.reshape(w_sitem_feature_T, (self.num_feature, 1))
                A, b = self._update_user_hyperparams(A, b, reward, w_sitem_feature)
                self.dic_userid_A_b[user_id] = (A, b)
            
            # compute RMSE
            print "iterations: %3d, train RMSE: %.6f" % (iter, sum_mse) 
            ret.append(str(iter) + ',' + str(sum_mse))
        return np.array(ret) 
    
    def _top_k_similarity(self, similarity_items, k):
        B = np.zeros(k, int)
        for i in xrange(k):
            idx = np.argmax(similarity_items)
            B[i]=idx
            similarity_items[idx]=-1        
        return B
    
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
    
def test(num_feature=30, delta=1, lamda_p=2, alpha=1.8):
    bmf_model = example()
    rating_matrix = bmf_model.matrix
    user_feature = bmf_model.user_features
    print 'user_feature.shape', len(user_feature)
    # required parameters
    beta_item = bmf_model.beta_item
    N = bmf_model.num_item
    # item features from MAP solution
#     item_features = bmf_model.item_features
#      
#     mat = np.matrix(item_features)
#     dataframe = pd.DataFrame(data=mat.astype(float))
#     dataframe.to_csv('../result/item_feature.csv', sep=',', header=False, float_format='%.2f', index=False)
    df = pd.read_csv('../result/item_feature.csv', sep=',', header = None)
    item_features = np.array(df)
    
    # split data to training & validation
    num_user, num_item, ratings = build_ml_len()
    matrix = build_rating_matrix(num_user, num_item, ratings)
    train_matrix = train_rating120(matrix)  
#     test = train_matrix[0]
#     ratings = []
#     for item in xrange(len(test)):
#          rating = [0 for x in xrange(3)]
#          if test[item] != 0:
#              rating[1] = item
#              rating[2] = test[item]
#              rating = np.array(rating)
#              ratings.append(rating)
#     ratings = np.array(ratings)
   
    UCB_Dependence_model = LinearUCBDependence(rating_matrix, user_feature, num_feature, item_features, delta, lamda_p, alpha, ratings) 
    
    result = UCB_Dependence_model._estimate(ratings)
    UCB_Dependence_model = open('../result/ret_linear_UCB_dependence', 'w')
    for item in result:
        print>>UCB_Dependence_model, item
    UCB_Dependence_model.close()    

if __name__ == "__main__":
    test()