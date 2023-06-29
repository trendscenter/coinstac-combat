#!/usr/bin/python

import sys
import json
import numpy as np
from numpy.core.numeric import argwhere
import combat

def list_recursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in list_recursive(v, key):
                yield found
        if k == key:
            yield v

def remote_0(args):
    input_list = args["input"]
    site_ids = list(input_list.keys())
    
    site_covar_list = [
        '{}_{}'.format('site', label) for index, label in enumerate(sorted(site_ids))    
    ]

    output_dict = {
        "site_covar_list": sorted(site_covar_list),
        "computation_phase": "remote_0"
    }

    cache_dict = {}

    computation_output = {"output": output_dict, "cache": cache_dict}
    
    return json.dumps(computation_output)

def remote_1(args):
    input_list = (args["input"])
    
    
    beta_vector_0 = [ np.array(input_list[site]["XtransposeX_local"]) for site in sorted(input_list.keys())]
    
    beta_vector_1 = sum(beta_vector_0)
    
    all_lambdas = [input_list[site]["lambda_value"] for site in sorted(input_list.keys())]
    if np.unique(all_lambdas).shape[0] != 1:
        raise Exception("Unequal lambdas at local sites")
    
    beta_vector_1 = beta_vector_1 + np.unique(all_lambdas) * np.eye(beta_vector_1.shape[0])   

    beta_vectors = np.matrix.transpose(
    sum([
        np.matmul(np.linalg.inv(beta_vector_1),
                    input_list[site]["Xtransposey_local"])
        for site in input_list
    ]))
    B_hat = beta_vectors.T
    
    n_batch =  len(input_list)
    
    sample_per_batch = np.array([ input_list[site]["local_sample_count"] for site in sorted(input_list.keys())])

    n_sample = sum(input_list[site]["local_sample_count"] for site in input_list)
    
    site_array = []
    for site in input_list:
        site_array = np.concatenate((site_array, [int(input_list[site]["site_index"])]*int(input_list[site]["local_sample_count"])), axis=0)
    
    grand_mean = np.dot((sample_per_batch/ float(n_sample)).T, B_hat[-n_batch:,:])
    # raise Exception(grand_mean, grand_mean.shape)
    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, n_sample)))
    
    output_dict = {
        "n_batch": n_batch,
        "B_hat": B_hat.tolist(),
        "n_sample": n_sample, 
        "grand_mean": grand_mean.tolist(),
        "stand_mean": stand_mean.tolist(),
        "site_array": site_array.tolist(),
        "computation_phase": "remote_1"
    }

    cache_dict = {
        "avg_beta_vector": B_hat.tolist(),
        "stand_mean": stand_mean.tolist(),
         "grand_mean": grand_mean.tolist()
    }

    computation_output = {"output": output_dict, "cache": cache_dict}
    return json.dumps(computation_output)

def remote_2(args):
    input_list = args["input"]
    var_pooled = [ np.array(input_list[site]["local_var_pooled"]) for site in sorted(input_list.keys())]
    global_var_pooled= sum(var_pooled)
    output_dict = {
        "global_var_pooled": global_var_pooled.tolist(),
        "computation_phase": "remote_2"
    }
    computation_output = {"output": output_dict}
    return json.dumps(computation_output)

def remote_3(args):
    output_dict = {"status": "Complete"}
   
    computation_output = { "output": output_dict, "success": True} 
    
    return json.dumps(computation_output) 

if __name__ == '__main__':
    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))
    if "local_0" in phase_key:
        computation_output = remote_0(parsed_args)
        sys.stdout.write(computation_output)
    elif "local_1" in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output) 
    elif  "local_2" in phase_key:
         computation_output = remote_2(parsed_args)
         sys.stdout.write(computation_output)        
    elif  "local_3" in phase_key:
        computation_output = remote_3(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")