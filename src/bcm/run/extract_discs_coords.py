import argparse
import os
import numpy as np

from bcm.methods.test_sct_label_vertebrae import test_sct_label_vertebrae
from bcm.methods.test_hourglass_network import test_hourglass
from bcm.methods.test_spinenet_network import test_spinenet
from bcm.methods.add_gt_coordinates import add_gt_coordinate_to_txt_file
from bcm.utils.init_txt_file import init_txt_file
from bcm.utils.utils import CONTRAST

def parser_default(args):
    '''
    This functions configure custom default values
    '''
    if args.out_txt_file == '':
        # Initialise output paths and name
        args.out_txt_file = os.path.abspath(os.path.join('results/files', f'{args.config_data["CONTRASTS"]}_discs_coords.txt'))
    return args

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract discs coords from benchmark methods')

    ## Parameters
    # All mandatory                          
    parser.add_argument('--config-data', type=str, metavar='<folder>',
                        help='Config JSON file where every label/image used for TESTING has its path specified ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--config-hg', type=str, required=True,
                        help='Config file where hourglass training parameters are stored Example: Example: ~/<your_path>/config.json (Required)')  # Hourglass config file
    
    # All methods
    parser.add_argument('-txt', '--out-txt-file', default='',
                        type=str, metavar='N',help='Generated txt file path (default="results/files/(CONTRAST)_discs_coords.txt")')
    parser.add_argument('--suffix-seg', type=str, default='_seg-manual',
                        help='Specify segmentation label suffix example: sub-296085_T2w(SEG_SUFFIX).nii.gz (default= "_seg")')
    parser.add_argument('--seg-folder', type=str, default='results',
                        help='Path to segmentation folder where non existing segmentations will be created. ' 
                        'These segmentations will be used to project labels onto the spinalcord (default="results")')
    
    args = parser_default(parser.parse_args())
    
    init_txt_file(args, split='TESTING', init_discs=11)
    add_gt_coordinate_to_txt_file(args)
    test_sct_label_vertebrae(args)
    test_spinenet(args)
    test_hourglass(args)
    
    print('All the methods have been computed')