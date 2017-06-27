'''
Created on Nov 13, 2016

@author: qingwang
'''
import numpy as np
import matplotlib.pyplot as plt


def draw_line():
    with open('../result/ret_random_sampling') as f_rand:
        rand_lines = f_rand.readlines()
    
    with open('../result/ret_thompson_sampling') as f_ts:
        ts_lines = f_ts.readlines()
    
    values = []
    for id in xrange(len(ts_lines)):
        value = ts_lines[id].split(',')[1]
        values.append(value)
        
    with open('../result/ret_cluster_thompson_sampling') as f_tsc:
        tsc_lines = f_tsc.readlines()
    
    tsc_values = []
    for id in xrange(len(tsc_lines)):
        value = tsc_lines[id].split(',')[1]
        tsc_values.append(value)    
     
    with open('../result/ret_linear_UCB') as f_ucb:
        ucb_lines = f_ucb.readlines()
    
    ucb_values = []
    for id in xrange(len(ucb_lines)):
        value = ucb_lines[id].split(',')[1]
        ucb_values.append(value)  
    
    with open('../result/ret_cluster_Linear_UCB') as f_ucbc:
        ucbc_lines = f_ucbc.readlines()
    
    ucbc_values = []
    for id in xrange(len(ucbc_lines)):
        value = ucbc_lines[id].split(',')[1]
        ucbc_values.append(value)  
        
        
    with open('../result/ret_epsilon_greedy_0.05') as f_epsilon:
        epsilon_lines = f_epsilon.readlines()
    
    epsilon_values005 = []
    for id in xrange(len(epsilon_lines)):
        value = epsilon_lines[id].split(',')[1]
        epsilon_values005.append(value) 
        
    with open('../result/ret_epsilon_greedy_0.1') as f_epsilon:
        epsilon_lines = f_epsilon.readlines()
    
    epsilon_values01 = []
    for id in xrange(len(epsilon_lines)):
        value = epsilon_lines[id].split(',')[1]
        epsilon_values01.append(value)        
    
    with open('../result/ret_epsilon_greedy_0.2') as f_epsilon:
        epsilon_lines = f_epsilon.readlines()
    
    epsilon_values02 = []
    for id in xrange(len(epsilon_lines)):
        value = epsilon_lines[id].split(',')[1]
        epsilon_values02.append(value)  
    
    with open('../result/ret_epsilon_greedy_0.5') as f_epsilon:
        epsilon_lines = f_epsilon.readlines()
    
    epsilon_values05 = []
    for id in xrange(len(epsilon_lines)):
        value = epsilon_lines[id].split(',')[1]
        epsilon_values05.append(value)
    
    
    
    with open('../result/ret_linear_UCB_dependence') as f_ucb_dependence:
        ucb_d_lines = f_ucb_dependence.readlines()
    
    ucb_d_values = []
    for id in xrange(len(ucb_d_lines)):
        value = ucb_d_lines[id].split(',')[1]
        ucb_d_values.append(value)
    
    
    x = np.arange(100000)
    plt.plot(x, rand_lines,label='Random')
    plt.plot(x, values, label='ICF-TS')
    plt.plot(x, tsc_values, label='ICF-TSC')
    plt.plot(x, ucb_values, label='ICF-UCB')
    plt.plot(x, ucbc_values, label='ICF-UCBC')
    plt.plot(x, ucb_d_values, label='ICF-UCB-DEPENENCE')
#     plt.plot(x, epsilon_values005, label='ICF-EPSILON-0.05')
    plt.plot(x, epsilon_values01, label='ICF-EPSILON-0.1')
#     plt.plot(x, epsilon_values02, label='ICF-EPSILON-0.2')
#     plt.plot(x, epsilon_values05, label='ICF-EPSILON-0.5')
    plt.ylabel('Cumulative MSE')
    plt.xlabel('Iteration')
    
    legend = plt.legend(loc='upper center', shadow=True)

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    
    plt.show()
   
if __name__ == '__main__':
    draw_line()