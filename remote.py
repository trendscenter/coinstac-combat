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




if __name__ == '__main__':
    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))
    if "local_0" in phase_key:
        computation_output = remote_0(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")