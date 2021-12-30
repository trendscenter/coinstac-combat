#!/usr/bin/python

import sys
import json
import numpy as np
import scipy.io as sp

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
        "site_covar_list": site_covar_list,
        "computation_phase": "remote_0"
    }

    cache_dict = {}

    computation_output = {"output": output_dict, "cache": cache_dict}
    
    return json.dumps(computation_output)

def remote_1(args):
    input_list = args["input"]
    beta_vector_0 = [ np.array(input_list[site]["XtransposeX_local"]) for site in input_list]
    beta_vector_1 = sum(beta_vector_0)
    all_lambdas = [input_list[site]["lambda_value"] for site in input_list]
    if np.unique(all_lambdas).shape[0] != 1:
        raise Exception("Unequal lambdas at local sites")
    
    beta_vector_1 = beta_vector_1 + np.unique(all_lambdas) * np.eye(beta_vector_1.shape[0])   
    avg_beta_vector = np.matrix.transpose(
    sum([
        np.matmul(np.linalg.inv(beta_vector_1),
                    input_list[site]["Xtransposey_local"])
        for site in input_list
    ]))


    total_sample_count = sum(input_list[site]["local_sample_count"] for site in input_list)

    
    output_dict = {
        "avg_beta_vector": avg_beta_vector.tolist(),
        "total_count": total_sample_count, 
        "computation_phase": "remote_1"
    }

    cache_dict = {
        "avg_beta_vector": avg_beta_vector.tolist()
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