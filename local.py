#!/usr/bin/python

import sys
import json
import scipy.io
import numpy as np
import combat
import csv
from local_ancillary import (add_site_covariates)

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
                line_count += 1
    return  data_url, lambda_value, covar_url

def local_0(args):
    input_list = args["input"]
    datapath = args["state"]["baseDirectory"] + "/" +  input_list["data"]
    data_url, lambda_value, covar_url = csv_parser(datapath)
    data_urls = args["state"]["baseDirectory"] + "/" +  data_url
    covar_urls = args["state"]["baseDirectory"] + "/" +  covar_url
    output_dict = {"computation_phase": "local_0"}
    cache_dict = {"data_urls": data_urls, "lambda_value": lambda_value, "covar_urls": covar_urls}
    computation_output = {"output": output_dict, "cache": cache_dict}
    return json.dumps(computation_output)

def local_1(args):
    covar_url =  args["cache"]["covar_urls"]
    data_url = args["cache"]["data_urls"]
    lambda_value = args["cache"]["lambda_value"]
    mat_X = scipy.io.loadmat(covar_url)
    mat_Y = scipy.io.loadmat(data_url)
    
    
    X = mat_X['mod']
    Y = mat_Y['data']
    sample_count = len(Y)
    
    augmented_X = add_site_covariates(args, X)
    biased_X = augmented_X.values
    XtransposeX_local = np.matmul(np.matrix.transpose(biased_X), biased_X)
    Xtransposey_local = np.matmul(np.matrix.transpose(biased_X), Y)
    output_dict = {
        "local_sample_count": sample_count,
        "XtransposeX_local": XtransposeX_local.tolist(),
        "Xtransposey_local": Xtransposey_local.tolist(),
        "lambda_value": lambda_value,
        "computation_phase": "local_1",
    }
    cache_dict = {
        "covariates": augmented_X.to_json(orient='split'),
    }
    computation_output = {"output": output_dict, "cache": cache_dict}

    return json.dumps(computation_output)
        
def local_2(args):
    #ComBat Method starts
    input_list = args["input"]
    cache_list = args["cache"]
    beta_matrix = np.array(input_list["avg_beta_vector"])
    cache_dict = {"beta_matrix": beta_matrix }
    output_dict = {

       "computation_phase": "local_2" 
    }

    computation_output = {"output": output_dict, "cache": cache_dict}

    return json.dumps(computation_output)
    raise Exception(beta_matrix.shape)
 
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
    else:
        raise ValueError("Error occurred at Local")
