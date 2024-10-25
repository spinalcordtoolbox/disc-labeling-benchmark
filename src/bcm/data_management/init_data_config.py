"""
Script based on https://github.com/spinalcordtoolbox/disc-labeling-hourglass
"""

import os
import argparse
import random
import json
import itertools
import numpy as np

from bcm.utils.utils import get_img_path_from_label_path, get_cont_path_from_other_cont, fetch_contrast, get_seg_path_from_label_path


# Determine specified contrasts
def init_data_config(args): 
    """
    Create a JSON configuration file from a TXT file where images paths are specified
    """
    if (args.split_validation + args.split_test) > 1:
        raise ValueError("The sum of the ratio between testing and validation cannot exceed 1")

    # Get input paths, could be label files or image files,
    # and make sure they all exist.
    file_paths = [os.path.abspath(path.replace('\n', '')) for path in open(args.txt)]
    if args.type == 'LABEL':
        label_paths = file_paths
        img_paths = [get_img_path_from_label_path(lp) for lp in label_paths]
        file_paths = label_paths + img_paths
    elif args.type == 'IMAGE':
        img_paths = file_paths
    elif args.type == 'CONTRAST-SC':
        if not args.cont: # If the target contrast is not specified
            raise ValueError(f'When using the type CONTRAST-SC, please specify the target contrast using the flag "--cont"')
        target_contrast = args.cont
        label_paths_input = file_paths
        label_paths_target = [get_cont_path_from_other_cont(lp, target_contrast) for lp in label_paths_input]
        img_paths_input = [get_img_path_from_label_path(lp) for lp in label_paths_input]
        img_paths_target = [get_img_path_from_label_path(lp) for lp in label_paths_target]
        file_paths = label_paths_input + label_paths_target + img_paths_input + img_paths_target
    elif args.type == 'CONTRAST':
        if not args.cont: # If the target contrast is not specified
            raise ValueError(f'When using the type CONTRAST, please specify the target contrast using the flag "--cont"')
        target_contrast = args.cont
        img_paths_input = file_paths
        img_paths_target = [get_cont_path_from_other_cont(ip, target_contrast) for ip in img_paths_input]
        file_paths = img_paths_input + img_paths_target
    elif args.type == 'LABEL-SEG':
        seg_suffix = args.suffix_seg
        if not seg_suffix:
            raise ValueError(f'When using the type LABEL-SEG, please specify the suffix of the associated segmentation using the flag "--suffix-seg"')
        label_paths = file_paths
        img_paths = [get_img_path_from_label_path(lp) for lp in label_paths]
        seg_paths = [get_seg_path_from_label_path(lp, seg_suffix=seg_suffix) for lp in label_paths]
        file_paths = label_paths + img_paths + seg_paths
    else:
        raise ValueError(f"invalid args.type: {args.type}")
    missing_paths = [
        path for path in file_paths
        if not os.path.isfile(path)
    ]
    
    if missing_paths:
        raise ValueError("missing files:\n" + '\n'.join(sorted(missing_paths)))
    
    # Extract BIDS parent folder path
    dataset_parent_path_list = ['/'.join(path.split('/sub')[0].split('/derivatives')[0].split('/')[:-1]) for path in file_paths]

    # Check if all the BIDS folders are stored inside the same parent repository
    if (np.array(dataset_parent_path_list) == dataset_parent_path_list[0]).all():
        dataset_parent_path = dataset_parent_path_list[0]
    else:
        raise ValueError('Please store all the BIDS datasets inside the same parent folder !')

    # Look up the right code for the set of contrasts present
    contrasts = "_".join(tuple(sorted(set(map(fetch_contrast, img_paths)))))

    config = {
        'TYPE': args.type,
        'CONTRASTS': contrasts,
        'DATASETS_PATH': dataset_parent_path
    }

    # Split into training, validation, and testing sets
    split_ratio = (1 - (args.split_validation + args.split_test), args.split_validation, args.split_test) # TRAIN, VALIDATION, and TEST
    
    if args.type == 'LABEL':
        config_paths = []
        for lp, ip in zip(label_paths, img_paths):
            config_paths.append({
                'IMAGE':ip.split(dataset_parent_path + '/')[-1], # Remove DATASETS_PATH
                'LABEL':lp.split(dataset_parent_path + '/')[-1]
            })
    elif args.type == 'CONTRAST-SC':
        config_paths = []
        for img_path_input, img_path_target, label_path_input, label_path_target in zip(img_paths_input, img_paths_target, label_paths_input, label_paths_target):
            config_paths.append({
                'INPUT_IMAGE':img_path_input.split(dataset_parent_path + '/')[-1], # Remove DATASETS_PATH
                'INPUT_LABEL':label_path_input.split(dataset_parent_path + '/')[-1],
                'TARGET_IMAGE':img_path_target.split(dataset_parent_path + '/')[-1],
                'TARGET_LABEL':label_path_target.split(dataset_parent_path + '/')[-1],
            })
    elif args.type == 'CONTRAST':
        config_paths = []
        for img_path_input, img_path_target in zip(img_paths_input, img_paths_target):
            config_paths.append({
                'INPUT_IMAGE':img_path_input.split(dataset_parent_path + '/')[-1], # Remove DATASETS_PATH
                'TARGET_IMAGE':img_path_target.split(dataset_parent_path + '/')[-1],
            })
    elif args.type == 'LABEL-SEG':
        config_paths = []
        for lp, ip, sp in zip(label_paths, img_paths, seg_paths):
            config_paths.append({
                'IMAGE':ip.split(dataset_parent_path + '/')[-1], # Remove DATASETS_PATH
                'LABEL':lp.split(dataset_parent_path + '/')[-1],
                'SEG':sp.split(dataset_parent_path + '/')[-1]
            })

    else:
        config_paths = [{'IMAGE':path.split(dataset_parent_path + '/')[-1]} for path in img_paths] # Remove DATASETS_PATH
    
    random.shuffle(config_paths)
    splits = [0] + [
        round(len(config_paths) * ratio)
        for ratio in itertools.accumulate(split_ratio)
    ]
    for key, (begin, end) in zip(
        ['TRAINING', 'VALIDATION', 'TESTING'],
        pairwise(splits),
    ):
        config[key] = config_paths[begin:end]

    # Save the config
    config_path = args.txt.replace('.txt', '') + '.json'
    json.dump(config, open(config_path, 'w'), indent=4)

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    # based on https://docs.python.org/3.11/library/itertools.html
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create config JSON from a TXT file which contains list of paths')
    
    ## Parameters
    parser.add_argument('--txt', required=True,
                        help='Path to TXT file that contains only image or label paths. (Required)')
    parser.add_argument('--type', choices=('LABEL', 'IMAGE', 'LABEL-SEG', 'CONTRAST-SC', 'CONTRAST'),
                        help='Type of paths specified. Choices are "LABEL", "IMAGE", "LABEL-SEG", "CONTRAST-SC" or "CONTRAST". (Required)')
    parser.add_argument('--cont', type=str, default='',
                        help='If the type CONTRAST or CONTRAST-SC is selected, this variable specifies the wanted contrast for target.')
    parser.add_argument('--suffix-seg', type=str, default='',
                        help='If the type LABEL-SEG is selected, this variable specifies the suffix of the associated segmentation file. The segmentation must be stored insiade the same folder')
    parser.add_argument('--split-validation', type=float, default=0.1,
                        help='Split ratio for validation. Default=0.1')
    parser.add_argument('--split-test', type=float, default=0.1,
                        help='Split ratio for testing. Default=0.1')
    
    args = parser.parse_args()
    
    if args.split_test > 0.9:
        args.split_validation = 1 - args.split_test
    
    init_data_config(args)