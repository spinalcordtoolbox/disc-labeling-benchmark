import os
import argparse
from spinalcordtoolbox.image import Image
import numpy as np

def get_dataset_parameters(args):
    '''
    The aim of this function is to extract parameters from the input dataset such as:
    - Number of subject --> Potentially missing files
    - Images resolution
    - Images dimensions
    - Suffixes (image, segmentation, discs_labels...)
    - Contrasts available
    - Maximum number of discs present in the images
    - ...
    '''
    label_suffix = '_labels-disc-manual'
    contrast = 'T2w'
    data_folder = os.path.abspath(args.datapath)
    sub_folders = os.listdir(data_folder)
    nb_discs = []
    max_disc = 0
    distribution_list = [0]*15
    for root, dirs, files in os.walk(data_folder):
        level = root.replace(data_folder, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
    # for sub in sub_folders:
    #     if sub.startswith('sub'):
    #         data_path = os.path.join(data_folder, sub)
    #         files = os.listdir(data_path)
    #         # TEMPORARY CODE
    #         path_label = os.path.join(data_path, f'{sub}_{contrast}{label_suffix}.nii.gz')
    #         disc_list = np.array([list(coord) for coord in Image(path_label).change_orientation('RSP').getNonZeroCoordinates(sorting='value')])
    #         for disc in disc_list:
    #             distribution_list[disc[-1]-1] += 1 # To determine the data distribution
    #         nb_discs.append(len(disc_list))
    #         if len(disc_list) > max_disc:
    #             max_disc = len(disc_list)
    #             sub_max = sub
    print(f'The maximum number of discs in the dataset {os.path.basename(data_folder)} is {max(nb_discs)} with the subject {sub_max}')
    print(f'The distribution is {distribution_list}')
            # sub-amu01_T2w_labels-disc-manual.nii.gz



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract parameters from the input dataset')

    ## Parameters
    # All                          
    parser.add_argument('--datapath', type=str, metavar='<folder>',
                        help='Path to data folder generated using src/bcm/utils/gather_data.py Example: ~/<your_dataset>/vertebral_data (Required)')
    
    get_dataset_parameters(parser.parse_args())                         
    