#!/usr/bin/python

import sys
import json
import numpy as np

def list_recursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in list_recursive(v, key):
                yield found
        if k == key:
            yield v

def remote_0(args):
    input_list = args["input"]
    total = sum([np.array(input_list[site]["value"][0]) for site in input_list])
    output_dict = {"computation_phase": "remote_0" ,  "total_sample": int(total), "total_site": len(input_list)}  
    cache_dict = {
     "total_sample": int(total),
     "total_site": len(input_list)
    }
    computation_output = { "output": output_dict, "cache": cache_dict } 
    return json.dumps(computation_output)

def remote_1(args):
    input_list = args["input"]
    grand_mean = sum([np.array(input_list[site]["local_grand_mean"]) for site in input_list])
    grand_variance = sum([np.array(input_list[site]["var_pooled"]) for site in input_list]) 
    output_dict = {
        "computation_phase": "remote_1",
        "grand_mean": grand_mean.tolist(),
        "grand_variance": grand_variance.tolist(),
        "s_mean": np.shape(grand_mean),
        "s_var": np.shape(grand_variance),
        "total_sample": args["cache"]["total_sample"],
        "total_site": args["cache"]["total_site"]
    }
    computation_output = { "output": output_dict} 
    return json.dumps(computation_output)

def remote_2(args):
    computation_output = { "output": args['input'], "success": True} 
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
    elif "local_2" in phase_key:
        # raise Exception(parsed_args)
        computation_output = remote_2(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")