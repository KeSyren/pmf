'''
Created on Nov 14, 2016

@author: qingwang
'''

import numpy as np
import matplotlib.pyplot as plt
from util.load_data import build_ml_len, build_rating_matrix

def draw_sparsity_user():
    num_user, num_item, ratings = build_ml_len()
    matrix = build_rating_matrix(num_user, num_item, ratings)
    print matrix.shape
    
    x = np.arange(num_item)
      
    plt.plot(x, matrix[6], 'k.')
    plt.axis([0,1000,1,6])
      
    plt.show()
#     matrix = matrix.T
#     x = np.arange(num_user)
#     plt.plot(x, matrix[0], 'b.')
#     plt.axis([0,num_user-1,0,6])
#     plt.show()
      

if __name__ == '__main__':
    draw_sparsity_user()