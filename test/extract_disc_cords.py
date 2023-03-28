import argparse
import os
import sys
from test_sct_label_vertebrae import test_sct_label_vertebrae
from test_hourglass_network import test_hourglass

parent_dir = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(parent_dir)

from utils.test_utils import CONTRAST


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract discs coords from sct and hourglass')

    ## Parameters
    parser.add_argument('--hg-datapath', type=str,
                        help='Hourglass dataset path')                               
    parser.add_argument('--sct-datapath', type=str,
                        help='SCT dataset path')                               
    parser.add_argument('-c', '--contrast', type=str, metavar='N', required=True,
                        help='MRI contrast')
    parser.add_argument('-txt', '--out-txt-file', type=str, metavar='N',
                        help='Generated txt file')
    parser.add_argument('--ndiscs', type=int, required=True,
                        help='Number of discs to detect')
    
    parser.add_argument('--att', default= True, type=bool,
                        help=' Use attention mechanism') 
    parser.add_argument('-s', '--stacks', default=2, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')                                                                                               
    
    args = parser.parse_args()
    
    contrast = CONTRAST[args.contrast]
    if args.out_txt_file is not None:
        txt_file = args.out_txt_file
    else:
        txt_file = os.path.join('files', f'{contrast}_hg{args.ndiscs}_discs_coords.txt')
        
    if not os.path.exists(txt_file):
        print("Creating txt file:", txt_file)
        with open(txt_file,"w") as f:
            f.write("subject_name contrast num_disc sct_discs_coords hourglass_coords gt_coords\n")

    test_sct_label_vertebrae(parser.parse_args())
    test_hourglass(parser.parse_args())