#!/usr/bin/python

import sys
import json
import scipy.io
import numpy as np
import combat

def list_recursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in list_recursive(v, key):
                yield found
        if k == key:
            yield v

def local_0(args):
    input_list = args["input"]
    index = input_list["index"]
    urls = args["state"]["baseDirectory"] + "/" +  input_list["url"]
    info = combat.send_sample_infomation(urls, index)
    output_dict = {"computation_phase": "local_0",  "value": info}
    cache_dict = {"url": urls, "location": index}
    computation_output = {"output": output_dict, "cache": cache_dict}
    return json.dumps(computation_output)

def local_1(args):
    agg_value = args["input"]["total_sample"]
    urls =  args["cache"]["url"]
    location = args["cache"]["location"]
    site_number = args["input"]["total_site"]
    _info_dict = {
    'n_batch': site_number - 1,
    'n_sample': agg_value
    }
    local_grand_mean, var_pooled = combat.local_grand_mean_with_betas(urls, _info_dict, 'batch',location)
    output_dict = {
        "computation_phase": "local_1",
        "local_grand_mean": local_grand_mean.tolist(),
        "var_pooled": var_pooled.tolist(),
        "s_mean": np.shape(local_grand_mean),
        "s_var": np.shape(var_pooled)
    }
    
    computation_output = {"output": output_dict }
    return json.dumps(computation_output)   

def local_2(args):
    agg_value = args["input"]["total_sample"]
    urls =  args["cache"]["url"]
    location = args["cache"]["location"]
    site_number = args["input"]["total_site"]
    _info_dict = {
    'n_batch': site_number - 1,
    'n_sample': agg_value
    }
    _grand_mean = np.array(args["input"]["grand_mean"])
    _var_pooled = np.array(args["input"]["grand_variance"])
    info = combat.adjust_data(urls, _grand_mean, _var_pooled, _info_dict, 'batch', location)
    output_dict = {
        "computation_phase": "local_2",
        "a_prior": info['estimates']['a_prior'], 
        "b_prior": info['estimates']['b_prior'] , 
        "t2": info['estimates']['t2'].tolist()
    }
    computation_output = {"output": output_dict }
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
    else:
        raise ValueError("Error occurred at Local")
