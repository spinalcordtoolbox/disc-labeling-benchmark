import argparse
import os
from test_sct_label_vertebrae import test_sct_label_vertebrae
from test_hourglass_network import test_hourglass


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract discs coords from sct and hourglass')

    ## Parameters
    parser.add_argument('--hg_datapath', type=str,
                        help='Hourglass dataset path')                               
    parser.add_argument('--sct_datapath', type=str,
                        help='SCT dataset path')                               
    parser.add_argument('-c', '--contrast', type=str, metavar='N', required=True,
                        help='MRI contrast')
    parser.add_argument('-txt', '--out_txt_file', default='visualize/discs_coords.txt', type=str, metavar='N',
                        help='Generated txt file')
    
    parser.add_argument('--att', default= True, type=bool,
                        help=' Use attention mechanism') 
    parser.add_argument('-s', '--stacks', default=2, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--njoints', default=11, type=int,
                        help='Number of joints')                                                                                               
    
    if not os.path.exists(parser.parse_args().out_txt_file):
        print("Creating txt file:", parser.parse_args().out_txt_file)
        with open(parser.parse_args().out_txt_file,"w") as f:
            f.write("subject_name contrast num_disc sct_discs_coords hourglass_coords gt_coords\n")

    #test_sct_label_vertebrae(parser.parse_args())
    test_hourglass(parser.parse_args())