
import numpy as np
import pandas as pd
import numpy.linalg as la
import math
import copy


def fit_LS_model_and_find_priors(input_list,cache_list):
    raise Exception(input_list.keys(), cache_list.keys())
    gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)    
    delta_hat = []
    for i, batch_idxs in enumerate(batch_info):
        delta_hat.append(np.var(s_data[:,batch_idxs],axis=1,ddof=1))
    delta_hat = list(map(convert_zeroes,delta_hat))
    gamma_bar = np.mean(gamma_hat, axis=1) 
    t2 = np.var(gamma_hat,axis=1, ddof=1)
    a_prior = list(map(aprior, delta_hat))
    b_prior = list(map(bprior, delta_hat))    
    
    LS_dict = {}
    LS_dict['gamma_hat'] = gamma_hat
    LS_dict['delta_hat'] = delta_hat
    LS_dict['gamma_bar'] = gamma_bar
    LS_dict['t2'] = t2
    LS_dict['a_prior'] = a_prior
    LS_dict['b_prior'] = b_prior
    return LS_dict
