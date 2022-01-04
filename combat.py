
import numpy as np
import pandas as pd
import numpy.linalg as la
import math
import copy
import scipy.io


def standarize_data(beta_matrix,input_list, cache_list):

    sample_per_batch = np.array(cache_list['local_sample_count'])
    n_sample = input_list['total_count']
    n_batch = input_list['n_batch']
    B_hat = np.array(input_list['avg_beta_vector'])
    
    grand_mean = np.dot((sample_per_batch/ float(n_sample)).T, B_hat[:n_batch,:])
    raise Exception(grand_mean)


0.02709778
0.02392642

0.25852891
0.18486775


-0.0339992 
-0.02653037