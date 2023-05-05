#===================================================
## Authors: 
# - Lucas Rouhier ()
# - Reza Azad (rezazad68@gmail.com)
# - Nathan Molinier (nathan.molinier@gmail.com)
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
#===================================================

import pickle
import argparse
import os

from dlh.utils.data2array import load_Data_Bids2Array_with_subjects
from dlh.utils.train_utils import extract_groundtruth_heatmap_with_subjects_and_GT_coords
from dlh.utils.test_utils import CONTRAST

def main(args):
    '''
    Prepare and split hourglass data into a trainset and a testset
    '''  
    contrast = CONTRAST[args.contrast]
    split_factor = args.split_factor
                   
    print('Loading dataset: ', os.path.abspath(args.datapath))
    ds_train = load_Data_Bids2Array_with_subjects(args.datapath, mode=contrast, factor=split_factor, split='train', aim='full')
    ds_test = load_Data_Bids2Array_with_subjects(args.datapath, mode=contrast, factor=split_factor, split='test', aim='full')  # we want to keep track of the subjects name
    
    print('Creating heatmap')
    full_train = extract_groundtruth_heatmap_with_subjects_and_GT_coords(ds_train)
    full_test = extract_groundtruth_heatmap_with_subjects_and_GT_coords(ds_test)  # we want to keep track of the subjects name and the ground truth position of the vertebral discs
    
    print('Saving the prepared datasets')
    prefix = args.prefix_name
    output_trainset = os.path.join(args.output_folder, prefix + '_train_' + contrast)
    with open(output_trainset, 'wb') as file_pi:
        pickle.dump(full_train, file_pi)
        
    output_testset = os.path.join(args.output_folder, prefix + '_test_' + contrast)
    with open(output_testset, 'wb') as file_pi:
        pickle.dump(full_test, file_pi)

    print('finished')   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare and split hourglass data into a trainset and a testset')
    
    ## Parameters
    parser.add_argument('--datapath', type=str, required=True,
                            help='Path to vertebral data')                     
    parser.add_argument('-o','--output-folder', type=str, required=True,
                            help='Path out to store the prepared datasets')  
    parser.add_argument('-c', '--contrast', type=str, required=True,
                            help='#t1_t2 for both t1 and t2 , t1 for t1 only , t2 for t2 only')
    parser.add_argument('--prefix-name', default='dataset', type=str,
                            help='Prefix particle in the dataset name (default="dataset")')
    parser.add_argument('--split-factor', default=0.9, type=float,
                            help='Split factor used for the training dataset')
                            
    main(parser.parse_args())