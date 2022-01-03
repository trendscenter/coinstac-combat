#!/usr/bin/python

import sys
import json
import numpy as np
import scipy.io as sp
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
        '{}_{}'.format('site', label) for index, label in enumerate(site_ids)
        
    ]


    output_dict = {
        "site_covar_list": sorted(site_covar_list),
        "computation_phase": "remote_0"
    }

    cache_dict = {}

    computation_output = {"output": output_dict, "cache": cache_dict}
    
    return json.dumps(computation_output)

def remote_1(args):
    input_list = args["input"]
    # raise Exception((sorted(input_list.keys())))
    beta_vector_0 = [ np.array(input_list[site]["XtransposeX_local"]) for site in (input_list)]
    beta_vector_1 = sum(beta_vector_0)
    # raise Exception(beta_vector_1)
    all_lambdas = [input_list[site]["lambda_value"] for site in input_list]
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
    sample_per_batch = np.array([ input_list[site]["local_sample_count"] for site in input_list])
    n_sample = sum(input_list[site]["local_sample_count"] for site in input_list)
    
    grand_mean = np.dot((sample_per_batch/ float(n_sample)).T, B_hat[:n_batch,:])
    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, n_sample)))

    output_dict = {
        "n_batch": n_batch,
        "avg_beta_vector": B_hat.tolist(),
        "total_count": n_sample, 
        "grand_mean": grand_mean.tolist(),
        "stand_mean": stand_mean.tolist(),
        "computation_phase": "remote_1"
    }

    cache_dict = {
        "avg_beta_vector": B_hat.tolist(),
        "stand_mean": stand_mean.tolist(),
         "grand_mean": grand_mean.tolist()
    }

    computation_output = {"output": output_dict, "cache": cache_dict}
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
    else:
        raise ValueError("Error occurred at Remote")