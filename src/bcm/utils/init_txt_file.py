import os
import argparse

from bcm.utils.utils import CONTRAST

def init_txt_file(args):
    datapath = os.path.abspath(args.datapath)
    contrast = CONTRAST[args.contrast][0]
    nb_discs_init = 11
    txt_file = args.out_txt_file
    methods_str = 'subject_name contrast num_disc gt_coords sct_discs_coords spinenet_coords hourglass_t1_coords hourglass_t2_coords hourglass_t1_t2_coords\n'
        
    if not os.path.exists(txt_file):
        os.makedirs(os.path.dirname(txt_file), exist_ok=True)
        print("Creating txt file:", txt_file)
        with open(txt_file,"w") as f:
            f.write(methods_str)
        
        # Initialize txt_file with subject_names and nb_discs_init
        print(f"Initializing txt file with subjects and {nb_discs_init} discs")
        for subject_name in os.listdir(datapath):
            if subject_name.startswith('sub'):
                # line = subject_name contrast disc_num ground_truth_coord + methods...
                subject_lines = [subject_name + ' ' + contrast + ' ' + str(disc_num + 1) + ' None'*(len(methods_str.split(' '))-3) + '\n' for disc_num in range(nb_discs_init)]
            with open(txt_file,"a") as f:
                f.writelines(subject_lines)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Init text file for benchmark')

    ## Parameters
    # All mandatory                          
    parser.add_argument('--datapath', type=str, metavar='<folder>',
                        help='Path to data folder generated using src/bcm/utils/gather_data.py Example: ~/<your_dataset>/vertebral_data (Required)')                               
    parser.add_argument('-c', '--contrast', type=str, required=True,
                        help='MRI contrast: choices=["t1", "t2"] (Required)')
    parser.add_argument('-txt', '--out-txt-file', default='',
                        type=str, metavar='N',help='Generated txt file path (default="results/files/(datapath_basename)_(CONTRAST)_discs_coords.txt")')
    
    # Run init_txt_file
    init_txt_file(parser.parse_args())