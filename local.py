#!/usr/bin/python

import sys
import json
import scipy.io
import numpy as np
import numpy.linalg as la
import combat
import csv
import pandas as pd
from local_ancillary import (add_site_covariates)
import copy
import math


def convert_zeroes(x):
    x[x==0] = 1
    return x
def aprior(delta_hat):
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat,ddof=1)
    return (2 * s2 +m**2) / float(s2)

def bprior(delta_hat):
    m = delta_hat.mean()
    s2 = np.var(delta_hat,ddof=1)
    return (m*s2+m**3)/s2

def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2*n*g_hat+d_star * g_bar) / (t2*n+d_star)

def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

def convert_zeroes(x):
    x[x==0] = 1
    return x

def int_eprior(sdat, g_hat, d_hat):
    r = sdat.shape[0]
    gamma_star, delta_star = [], []
    for i in range(0,r,1):
        g = np.delete(g_hat,i)
        d = np.delete(d_hat,i)
        x = sdat[i,:]
        n = x.shape[0]
        j = np.repeat(1,n)
        A = np.repeat(x, g.shape[0])
        A = A.reshape(n,g.shape[0])
        A = np.transpose(A)
        B = np.repeat(g, n)
        B = B.reshape(g.shape[0],n)
        resid2 = np.square(A-B)
        sum2 = resid2.dot(j)
        LH = 1/(2*math.pi*d)**(n/2)*np.exp(-sum2/(2*d))
        LH = np.nan_to_num(LH)
        gamma_star.append(sum(g*LH)/sum(LH))
        delta_star.append(sum(d*LH)/sum(LH))
    adjust = (gamma_star, delta_star)
    return adjust


def find_non_parametric_adjustments(s_data, LS):
    gamma_star, delta_star = [], []
    temp = int_eprior(s_data, LS['gamma_hat'], LS['delta_hat'])
    gamma_star.append(temp[0])
    delta_star.append(temp[1])
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)
    # raise Exception(gamma_star.shape, delta_star.shape)
    return gamma_star, delta_star


def adjust_data_final(s_data, batch_design, gamma_star, delta_star, stand_mean, mod_mean, var_pooled, dat, local_n_sample):
    bayesdata = s_data
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)
    dsq = np.sqrt(delta_star)
    # dsq = dsq.reshape((len(dsq), 1))
    denom = np.dot(dsq.T, np.ones((1, local_n_sample)))
    numer = np.array(bayesdata  - np.dot(batch_design, gamma_star).T)
    bayesdata = numer / denom
    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))

    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, local_n_sample))) + stand_mean + mod_mean
    return bayesdata





def list_recursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in list_recursive(v, key):
                yield found
        if k == key:
            yield v

def csv_parser(file_url):
    with open(file_url, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                data_url = row[0]
                lambda_value = int(row[1])
                covar_url= row[2]
                site_index = row[3]
                line_count += 1
    return  data_url, lambda_value, covar_url, site_index

def local_0(args):
    input_list = args["input"]
    datapath = args["state"]["baseDirectory"] + "/" +  input_list["data"]
    data_url, lambda_value, covar_url, site_index = csv_parser(datapath)
    data_urls = args["state"]["baseDirectory"] + "/" +  data_url
    covar_urls = args["state"]["baseDirectory"] + "/" +  covar_url
    output_dict = {"computation_phase": "local_0"}
    cache_dict = {"data_urls": data_urls, "lambda_value": lambda_value, "covar_urls": covar_urls, "site_index": site_index}
    computation_output = {"output": output_dict, "cache": cache_dict}
    return json.dumps(computation_output)

def local_1(args):
    # raise Exception( args["cache"])
    covar_url =  args["cache"]["covar_urls"]
    data_url = args["cache"]["data_urls"]
    lambda_value = args["cache"]["lambda_value"]
    mat_X = scipy.io.loadmat(covar_url)
    mat_Y = scipy.io.loadmat(data_url)
    
    site_index = args["cache"]["site_index"]
    X = mat_X['mod']
    Y = mat_Y['data']
    sample_count = len(Y)
    
    augmented_X = add_site_covariates(args, X)
    biased_X = augmented_X.values
    # raise Exception(biased_X)
    XtransposeX_local = np.matmul(np.matrix.transpose(biased_X), biased_X)
    Xtransposey_local = np.matmul(np.matrix.transpose(biased_X), Y)
    output_dict = {
        "local_sample_count": sample_count,
        "XtransposeX_local": XtransposeX_local.tolist(),
        "Xtransposey_local": Xtransposey_local.tolist(),
        "lambda_value": lambda_value,
        "site_index": site_index,
        "computation_phase": "local_1",
    }
    cache_dict = {
        "covariates": augmented_X.to_json(orient='split'),
        "local_sample_count": sample_count
    }
    computation_output = {"output": output_dict, "cache": cache_dict}

    return json.dumps(computation_output)
        
def local_2(args):
    
    input_list = args["input"]
    cache_list = args["cache"]
    covar = pd.read_json(cache_list["covariates"], orient='split')
    mat_Y = scipy.io.loadmat(cache_list["data_urls"])
    data = mat_Y['data'].T
    design = covar.values
    B_hat = np.array(input_list["B_hat"])
    n_sample = input_list["n_sample"]
    n_batch = input_list["n_batch"]
    site_array = input_list["site_array"]
    local_n_sample = cache_list["local_sample_count"]
    local_var_pooled = np.dot(((data - np.dot(design, B_hat).T)**2), np.ones((local_n_sample, 1)) / float(n_sample))
    stand_mean = np.array(input_list["stand_mean"])
    mod_mean = 0
    if design is not None:
        tmp = copy.deepcopy(design)
        tmp[:,range(-n_batch,0)] = 0
        mod_mean = np.transpose(np.dot(tmp, B_hat))    

    # raise Exception(mod_mean.shape)

    cache_dict = {
        "mod_mean": mod_mean.tolist(),
        "stand_mean": stand_mean.tolist(),
        "site_array": site_array,
        "n_batch": n_batch

    }
    output_dict = {
       "local_var_pooled": local_var_pooled.tolist(),
       "computation_phase": "local_2" 
    }

    computation_output = {"output": output_dict, "cache": cache_dict}
   
    return json.dumps(computation_output)
    
 
def local_3(args):
    
    input_list = args["input"]
    cache_list = args["cache"]
    
    var_pooled = np.array(input_list["global_var_pooled"])
    mat_Y = scipy.io.loadmat(cache_list["data_urls"])
    data = mat_Y['data'].T
    stand_mean = np.array(cache_list["stand_mean"]).T
    # raise Exception(stand_mean[169], stand_mean.shape)
    mod_mean = np.array(cache_list["mod_mean"])
    local_n_sample = cache_list["local_sample_count"]

    site_index = cache_list["site_index"]
    site_array = cache_list["site_array"]
    # raise Exception(site_index, site_array)
    indices = [index for index, element in enumerate(site_array) if element == int(site_index)]
    # raise Exception(stand_mean[96,0], stand_mean.shape)
    filtered_mean = stand_mean[indices]
    # raise Exception(filtered_mean, filtered_mean.shape)
    local_stand_mean = filtered_mean.T
    s_data = ((data - local_stand_mean - mod_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, local_n_sample))))
    # raise Exception(data[0], cache_list["data_urls"], data.shape, local_stand_mean.shape, local_stand_mean[0])
    # raise Exception(data.shape, local_stand_mean.shape, mod_mean.shape, var_pooled.shape)
    # raise Exception(s_data.shape, s_data[0])


   
    # combat.fit_LS_model_and_find_priors(input_list,cache_list)
    # ComBat section starts


    covar = pd.read_json(cache_list["covariates"], orient='split')
    design = covar.values
    n_batch = cache_list["n_batch"]
    # batch_design = design[:,-1:]
    batch_design = np.array([[1]*local_n_sample]).T
    # raise Exception(batch_design.shape)
    gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)
    delta_hat = []
    delta_hat.append(np.var(s_data,axis=1,ddof=1))
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
    gamma_star, delta_star = find_non_parametric_adjustments(s_data, LS_dict)

    bayesdata = adjust_data_final(s_data, batch_design, gamma_star, delta_star, local_stand_mean, mod_mean, var_pooled,  data, local_n_sample)

    harmonized_data = np.transpose(bayesdata)
    output_url = args["state"]["outputDirectory"] + "/"
    np.savetxt(output_url + 'harmonized_site_'+ str(site_index) +'_data.csv', harmonized_data, delimiter=',') 

    # #######   comBat ends  ###############################
    output_dict = {
       "message": "Data Harmonization complete", 
       "computation_phase": "local_3" 
    }
    computation_output = {"output": output_dict}
    return json.dumps(computation_output)    

if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_0(parsed_args)
        sys.stdout.write(computation_output)
    elif "remote_0" in phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    elif "remote_1" in phase_key:
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    elif "remote_2" in phase_key:
        computation_output = local_3(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
