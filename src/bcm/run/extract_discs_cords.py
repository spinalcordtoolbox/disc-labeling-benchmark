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
        args.out_txt_file = os.path.abspath(os.path.join('results/files', f'{os.path.basename(os.path.normpath(args.datapath))}_{CONTRAST[args.contrast][0]}_discs_coords.txt'))
    return args

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract discs coords from benchmark methods')

    ## Parameters
    # All mandatory                          
    parser.add_argument('--datapath', type=str, metavar='<folder>',
                        help='Path to data folder generated using src/bcm/utils/gather_data.py Example: ~/<your_dataset>/vertebral_data (Required)')                               
    parser.add_argument('-c', '--contrast', type=str, required=True,
                        help='MRI contrast: choices=["t1", "t2"] (Required)')
    parser.add_argument('--config-hg', type=str, required=True,
                        help='Config file where hourglass training parameters are stored Example: Example: ~/<your_path>/config.json (Required)')  # Hourglass config file
    
    # All methods
    parser.add_argument('-txt', '--out-txt-file', default='',
                        type=str, metavar='N',help='Generated txt file path (default="results/files/(datapath_basename)_(CONTRAST)_discs_coords.txt")')
    parser.add_argument('--suffix-img', type=str, default='',
                        help='Specify img suffix example: sub-250791(IMG_SUFFIX)_T2w.nii.gz (default= "")')
    parser.add_argument('--suffix-label-disc', type=str, default='_labels-disc-manual',
                        help='Specify label suffix example: sub-250791(IMG_SUFFIX)_T2w(DISC_LABEL_SUFFIX).nii.gz (default= "_labels-disc-manual")')
    parser.add_argument('--suffix-seg', type=str, default='_seg',
                        help='Specify segmentation label suffix example: sub-296085(IMG_SUFFIX)_T2w(SEG_SUFFIX).nii.gz (default= "_seg")')
    args = parser_default(parser.parse_args())
    
    init_txt_file(args)
    add_gt_coordinate_to_txt_file(args)
    test_sct_label_vertebrae(args)
    test_spinenet(args)
    test_hourglass(args)
    
    print('All the methods have been computed')