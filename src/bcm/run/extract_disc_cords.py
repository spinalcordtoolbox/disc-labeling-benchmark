import argparse
import os
import numpy as np

from bcm.methods.test_sct_label_vertebrae import test_sct_label_vertebrae
from bcm.methods.test_hourglass_network import test_hourglass
from bcm.methods.test_spinenet_network import test_spinenet
from bcm.utils.utils import CONTRAST, swap_y_origin, project_on_spinal_cord, edit_subject_lines_txt_file

from spinalcordtoolbox.utils.sys import run_proc

from dlh.utils.data2array import mask2label, get_midNifti

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

def add_gt_coordinate_to_txt_file(args):
    '''
    Add ground truth coordinates to text file
    '''
    datapath = os.path.abspath(args.datapath)
    contrast = CONTRAST[args.contrast][0]
    txt_file = args.out_txt_file
    dir_list = os.listdir(datapath)
    label_suffix = args.suffix_label
    img_suffix = args.suffix_img
    seg_suffix = '_seg'
    
    # Load disc_coords txt file
    with open(txt_file,"r") as f:  # Checking already processed subjects from txt file
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
    
    print('Adding ground truth coords')
    for dir_name in dir_list:
        if dir_name.startswith('sub'):
            img_path = os.path.join(datapath, dir_name, dir_name + img_suffix + '_' + contrast + '.nii.gz')
            label_path = os.path.join(datapath, dir_name, dir_name + img_suffix + '_' + contrast + label_suffix + '.nii.gz')
            seg_path = os.path.join(datapath, dir_name, dir_name + img_suffix + '_' + contrast + seg_suffix + '.nii.gz' )
            if not os.path.exists(label_path):
                print(f'Error while importing {dir_name}\n {img_path} may not exist\n {label_path} may not exist, please check suffix {label_suffix}\n')
            else:
                if os.path.exists(seg_path):
                    status = 0
                else:
                    status, _ = run_proc(['sct_deepseg_sc',
                                            '-i', img_path, 
                                            '-c', args.contrast,
                                            '-o', seg_path])
                if status != 0:
                    print(f'Fail segmentation for {dir_name}')
                else:
                    img_shape = get_midNifti(img_path).shape
                    discs_labels = mask2label(label_path)
                    gt_coord = np.array(discs_labels)
                    
                    # Project on spinalcord
                    gt_coord = project_on_spinal_cord(coords=gt_coord, seg_path=seg_path, disc_num=True, proj_2d=False)
                    
                    # Remove thinkness coordinate
                    gt_coord = gt_coord[:, 1:]
                    
                    # Swap axis prediction and ground truth
                    gt_coord = swap_y_origin(coords=gt_coord, img_shape=img_shape, y_pos=0).astype(int)  # Move y origin to the bottom of the image like Niftii convention
                    
                    # Edit coordinates in txt file
                    # line = subject_name contrast disc_num gt_coords sct_discs_coords spinenet_coords hourglass_t1_coords hourglass_t2_coords hourglass_t1_t2_coords
                    split_lines = edit_subject_lines_txt_file(coords=gt_coord, txt_lines=split_lines, subject_name=dir_name, contrast=contrast, method_name='gt_coords')
    
    for num in range(len(split_lines)):
        split_lines[num] = ' '.join(split_lines[num])
        
    with open(txt_file,"w") as f:
        f.writelines(split_lines)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract discs coords from benchmark methods')

    ## Parameters
    # All                          
    parser.add_argument('--datapath', type=str, metavar='<folder>',
                        help='Path to data folder generated using bcm/utils/gather_data.py Example: ~/<your_dataset>/vertebral_data (Required)')                               
    parser.add_argument('-c', '--contrast', type=str, required=True,
                        help='MRI contrast: choices=["t1", "t2"] (Required)')
    parser.add_argument('-txt', '--out-txt-file', default=os.path.abspath(os.path.join('results/files', f'{os.path.basename(os.path.normpath(parser.parse_args().datapath))}_{CONTRAST[parser.parse_args().contrast][0]}_hg{parser.parse_args().ndiscs}_discs_coords.txt')),
                        type=str, metavar='N',help='Generated txt file path (default="results/files/(datapath_basename)_(test_CONTRAST)_hg(nb_class_hourglass)_discs_coords.txt")')
    
    parser.add_argument('--ndiscs', type=int, default=15,
                        help='Number of class hourglass (default=15)')
    parser.add_argument('--suffix-label', type=str, default='_labels-disc-manual',
                        help='Specify label suffix (default= "_labels-disc-manual")') 
    parser.add_argument('--suffix-img', type=str, default='',
                        help='Specify img suffix (default= "")')
    parser.add_argument('--skeleton-dir', default=os.path.join(parser.parse_args().datapath, 'skeletons'),
                        type=str, metavar='<folder>',help='Path to skeleton dir (default=<datapath>/skeletons)')
    parser.add_argument('--weights-dir', default='../disc-labeling-hourglass/src/dlh/weights',
                        type=str, metavar='<folder>',help='Path to weights folder hourglass (default=../disc-labeling-hourglass/src/dlh/weights)')
    parser.add_argument('--train-contrasts', default="all", type=str,
                        help='MRI contrast used for the hourglass training'
                        '(default= "all")'
                        'write "all" for multipe contrast comparison')
    parser.add_argument('--att', default=True, type=bool,
                        help=' Use attention mechanism (default=True)') 
    parser.add_argument('-s', '--stacks', default=2, type=int, metavar='N',
                        help='Number of hourglasses to stack (default=2)')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass (default=1)')                                                                                               
    
    init_txt_file(parser.parse_args())
    add_gt_coordinate_to_txt_file(parser.parse_args())
    test_sct_label_vertebrae(parser.parse_args())
    test_spinenet(parser.parse_args())
    test_hourglass(parser.parse_args())
    
    print('All the methods have been computed')