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
                index = int(row[1])
                covar_url= row[2]
                line_count += 1
    return  data_url, index, covar_url

def local_0(args):
    input_list = args["input"]
    datapath = args["state"]["baseDirectory"] + "/" +  input_list["data"]
    data_url, index, covar_url = csv_parser(datapath)
    data_urls = args["state"]["baseDirectory"] + "/" +  data_url
    covar_urls = args["state"]["baseDirectory"] + "/" +  covar_url
    output_dict = {"computation_phase": "local_0"}
    cache_dict = {"data_urls": data_urls, "location": index, "covar_urls": covar_urls}
    computation_output = {"output": output_dict, "cache": cache_dict}
    return json.dumps(computation_output)

def local_1(args):
    covar_url =  args["cache"]["covar_urls"]
    mat = scipy.io.loadmat(covar_url)
    X = mat['mod']
    augmented_X = add_site_covariates(args, X)
    raise Exception(augmented_X)

    

 
if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_0(parsed_args)
        sys.stdout.write(computation_output)
    elif "remote_0" in phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
