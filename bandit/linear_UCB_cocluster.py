'''
Created on Oct 23, 2016

@author: qingwang
'''
from mf.bayesian_matrix_factorization import example
import math
import random
import numpy as np
import numpy.random as rand
from numpy.linalg import inv, cholesky, norm
from util.load_data import build_ml_len, best_rating_user, build_rating_matrix, train_rating120
from scipy.spatial.distance import euclidean


class LinearUCBCoCluster():
    
    def __init__(self, num_item, user_feature, num_feature, item_features, delta, lamda_p, alpha, train):
        self.num_item = num_item
        self.num_feature = num_feature
        # user_id item_id rating 100,000 * 3
        self.train = train
        # user feature
        self.user_feature = user_feature
        
        # item features
        self.item_features = item_features
        
        # required user parameters
        self.dic_userid_A_b = {}
        self.dic_clusterid_M_n = {}
        self.dic_clusterid_users = {}
        self.dic_clusterid_user_feature = {}
        
        # required item parameters
        self.dic_itemid_C_d = {}
        self.dic_clusterid_O_p = {}
        self.dic_clusterid_items = {}
        self.dic_clusterid_item_feature = {}
        
        self.delta = delta
        self.lamda_p = lamda_p
        self.alpha = alpha
     
    def _init_parameters_cluster_user(self, num_cluster, num_user):
        # initialize user parameters
        A = np.eye(self.num_feature, dtype='float64') * self.lamda_p
        b = np.zeros((self.num_feature, 1), dtype='float64')
        
        for user_id in xrange(num_user):
            self.dic_userid_A_b[user_id] = (A, b)
            
            cluster_id = random.randint(0, num_cluster-1)
            if self.dic_clusterid_users.get(cluster_id) == None:
                membs = []
                membs.append(user_id)
                self.dic_clusterid_users[cluster_id] = membs
            else:
                membs = self.dic_clusterid_users.get(cluster_id)
                membs.append(user_id)
                self.dic_clusterid_users[cluster_id] = membs
                
        # initialize cluster parameters
        M = np.eye(self.num_feature, dtype='float64') * self.lamda_p
        n = np.zeros((self.num_feature, 1), dtype='float64') 
        for cluster_id in xrange(num_cluster):
            self.dic_clusterid_user_feature[cluster_id] = np.dot(inv(M), n)
            self.dic_clusterid_M_n[cluster_id] = (M, n)
          
    def _init_parameters_cluster_item(self, num_cluster, num_item):
        # initialize item parameters
        C = np.eye(self.num_feature, dtype='float64') * self.lamda_p
        d = np.zeros((self.num_feature, 1), dtype='float64')
        
        for item_id in xrange(num_item):
            self.dic_clusterid_C_d[item_id] = (C, d)
            
            cluster_id = random.randint(0, num_cluster-1)
            if self.dic_clusterid_items.get(cluster_id) == None:
                membs = []
                membs.append(item_id)
                self.dic_clusterid_items[cluster_id] = membs
            else:
                membs = self.dic_clusterid_items.get(cluster_id)
                membs.append(item_id)
                self.dic_clusterid_items[cluster_id] = membs
                
        # initialize cluster parameters
        O = np.eye(self.num_feature, dtype='float64') * self.lamda_p
        p = np.zeros((self.num_feature, 1), dtype='float64') 
        for cluster_id in xrange(num_cluster):
            self.dic_clusterid_item_feature[cluster_id] = np.dot(inv(O), p)
            self.dic_clusterid_O_p[cluster_id] = (O, p)
     
               
    def _estimate(self, train, tolerance=1e-5):
        # print train.shape
        sum_mse = 0
        iter = 0
        ret = []
        for id in xrange(len(train)):
            user_id = train[id][0] 
            user_cluster_id = self._find_user_clusterid(user_id)
            
            max_rating = best_rating_user(self.rating_matrix, user_id)
            iter += 1
            print 'user_id ', user_id 
            print 'user_cluster_id', user_cluster_id
            
            A = self.dic_userid_A_b.get(user_id)[0]
            b = self.dic_userid_A_b.get(user_id)[1]
            M = self.dic_clusterid_M_n.get(user_cluster_id)[0]
            n = self.dic_clusterid_M_n.get(user_cluster_id)[1]
            
            w_user_cluster_feature = self.dic_clusterid_user_feature.get(user_cluster_id)
            w_user_cluster_features_T = np.reshape(w_user_cluster_feature, (self.num_feature,))
            
            self.best_item_feature = np.zeros((self.num_feature, 1), dtype='float64')
            max_estimated_reward = -0.1
            best_item  = -1
            
            for item_id in xrange(self.num_item):
                item_cluster_id = self._find_item_clusterid(item_id)
                w_item_feature_T = self.item_features[item_id]
                w_item_feature = np.reshape(w_item_feature_T, (self.num_feature,1))
                
                confidence = math.sqrt(np.dot(np.dot(w_item_feature_T, inv(M) * pow(self.delta,2)), w_item_feature)*math.log(id+1))
                estimated_reward = np.dot(w_user_cluster_features_T, w_item_feature) + self.alpha * confidence
                
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
                print real_reward, max_estimated_reward
            #sum_mse += math.pow((real_reward-max_estimated_reward), 2)
            sum_mse += math.pow((real_reward - max_rating), 2)
            # update A, b parameters
            A, b = self._update_user_hyperparams(A, b, real_reward, self.best_item_feature)
            w_user_feature = np.dot(inv(A), b)
            self.dic_userid_A_b[user_id] = (A, b)
            self._update_user_cluster_members(w_user_feature, user_id, user_cluster_id)
            # compute RMSE
            print "iterations: %3d, train RMSE: %.6f" % (iter, sum_mse) 
            ret.append(str(iter) + ',' + str(sum_mse))
        return np.array(ret)      
    
    def _update_user_cluster_members(self, w_user_feature, cur_userid, cur_clusterid):
        w_user_feature_T = np.reshape(w_user_feature, (self.num_feature,))
        w_cur_cluster_feature_T = np.reshape(self.dic_clusterid_user_feature.get(cur_clusterid), (self.num_feature,))
        min_dis = euclidean(w_user_feature_T, w_cur_cluster_feature_T)
        # init updated cluster
        updated_clusterid = cur_clusterid
        
        for cluster_id in self.dic_clusterid_user_feature.keys():
            w_cluster_feature_T = np.reshape(self.dic_clusterid_user_feature.get(cluster_id), (self.num_feature,))
            distance_cluster = euclidean(w_user_feature_T, w_cluster_feature_T)
            if distance_cluster < min_dis:
                updated_clusterid = cluster_id
        
        # reassign user_id
        if updated_clusterid != cur_clusterid:
            self.dic_clusterid_users[cur_clusterid].remove(cur_userid)
            self.dic_clusterid_users[updated_clusterid].append(cur_userid)
            self._update_user_cluster_parameters(cur_clusterid)
            self._update_user_cluster_parameters(updated_clusterid)
        
        else:
            self._update_user_cluster_parameters(cur_clusterid)
            
    def _update_user_cluster_parameters(self, cur_clusterid):       
        I = np.eye(self.num_feature, dtype='float64') * self.lamda_p
        M = I
        n = np.zeros((self.num_feature, 1), dtype='float64') 
        
        for userid in self.dic_clusterid_users.get(cur_clusterid):
            A = self.dic_userid_A_b.get(userid)[0]
            b = self.dic_userid_A_b.get(userid)[1]
            M = np.add(M, np.subtract(A, I))
            n = np.add(n, b)
        
        self.dic_clusterid_M_n[cur_clusterid] = (M,n)  
        w_updated_cluster_feature = np.dot(inv(M), n)    
        self.dic_clusterid_user_feature[cur_clusterid] = w_updated_cluster_feature
    
    def _update_user_hyperparams(self, A, b, reward, best_item_feature):
        # _get_cluster_params mean and variance
        best_item_feature_T = np.reshape(best_item_feature, (self.num_feature,))
        A = A + np.outer(best_item_feature_T, best_item_feature)
        b = b + reward * best_item_feature
        return A, b
    
    def _find_user_clusterid(self, user_id):
        # find user belong to cluster
        for cluster_id in self.dic_clusterid_users.keys():
            if user_id in self.dic_clusterid_users[cluster_id]:
                return cluster_id
    
    def _find_item_clusterid(self, item_id):
        # find item belong to cluster
        for cluster_id in self.dic_clusterid_items.keys():
            if item_id in self.dic_clusterid_items[cluster_id]:
                return cluster_id
    
def test(num_feature=30, delta=1, lamda_p=2, alpha=1.8):
    bmf_model = example()
    rating_matrix = bmf_model.matrix
    user_feature = bmf_model.user_features
    print 'user_feature.shape', len(user_feature)
   
    # item mean and variance
    item_features = bmf_model.item_features

    num_user, num_item, ratings = build_ml_len()
    
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

    linearUCB_cocluster_model = LinearUCBCoCluster(num_item, user_feature, num_feature, item_features, delta, lamda_p, alpha, ratings) 
    linearUCB_cocluster_model._init_parameters_cluster_user(20, num_user)
    linearUCB_cocluster_model._init_parameters_cluster_item(20, num_item)
    
    result = linearUCB_cocluster_model._estimate(ratings)
    cocluster_linearUCB_file = open('../result/ret_cocluster_Linear_UCB_test', 'w')
    for item in result:
        cocluster_linearUCB_file.write(item + '\n')
    cocluster_linearUCB_file.close()    

if __name__ == "__main__":
    test()