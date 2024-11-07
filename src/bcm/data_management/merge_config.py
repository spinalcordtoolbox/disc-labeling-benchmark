import argparse
import json
import numpy as np
import os

def merge_json(args):
    # Load parameters
    json_paths = args.jsons
    out_json_path = args.out

    # Load first json
    with open(json_paths[0], "r") as file:
        out_config = json.load(file)

    # Open JSON configs
    keys = out_config.keys()
    for path in json_paths[1:]:
        # load JSON
        with open(path, "r") as file:
            config = json.load(file)
        
        # Check for issues between keys
        if keys != config.keys():
            raise ValueError(f'JSON keys do not match between {json_paths[0]} and {path}')
        
        # Loop on keys
        for key in keys:
            if isinstance(out_config[key], str):
                if key == "CONTRASTS":
                    contrast_list = list(np.unique(out_config[key].split('_') + config[key].split('_')))
                    out_config[key] = "_".join(contrast_list)
                else:
                    if out_config[key] != config[key]:
                        raise ValueError(f'Error: {out_config[key]} != {config[key]} {key} values should match')
            if isinstance(out_config[key], list):
                out_config[key]+=config[key]

    # Create missing directories
    if not os.path.exists(os.path.dirname(out_json_path)):
        os.makedirs(os.path.dirname(out_json_path))
    
    # Save output JSON file
    json.dump(out_config, open(out_json_path, 'w'), indent=4)

    






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create config JSON from a TXT file which contains list of paths')
    
    ## Parameters
    parser.add_argument('--jsons', type=str, nargs='+', required=True, 
                        help='Path to the JSON files that have to be merged (Required)')
    parser.add_argument('--out', type=str, required=True, 
                        help='Path to output JSON file after being merged (Required)')
    
    args = parser.parse_args()

    merge_json(args)