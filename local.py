#!/usr/bin/python

import sys
import json
import scipy.io
import numpy as np
import combat
import csv
import pandas as pd
from local_ancillary import (add_site_covariates)
import copy

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
        "site_array": site_array

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

    cache_dict = {
        "s_data": s_data.tolist()
    }
    output_dict = {
       "computation_phase": "local_3" 
    }

    computation_output = {"output": output_dict, "cache": cache_dict}
   
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
