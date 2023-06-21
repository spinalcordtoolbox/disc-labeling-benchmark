import os
import argparse
import numpy as np

from bcm.utils.utils import CONTRAST, swap_y_origin, project_on_spinal_cord, edit_subject_lines_txt_file
from bcm.utils.init_txt_file import init_txt_file

from spinalcordtoolbox.utils.sys import run_proc
from spinalcordtoolbox.image import Image

from dlh.utils.data2array import mask2label, get_midNifti

def add_gt_coordinate_to_txt_file(args):
    '''
    Add ground truth coordinates to text file
    '''
    datapath = os.path.abspath(args.datapath)
    contrast = CONTRAST[args.contrast][0]
    txt_file = args.out_txt_file
    dir_list = os.listdir(datapath)
    img_suffix = args.suffix_img
    disc_label_suffix = args.suffix_label_disc
    seg_suffix = args.suffix_seg
    
    # Load disc_coords txt file
    with open(txt_file,"r") as f:  # Checking already processed subjects from txt file
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
    
    print('Adding ground truth coords')
    for dir_name in dir_list:
        if dir_name.startswith('sub'):
            img_path = os.path.join(datapath, dir_name, dir_name + img_suffix + '_' + contrast + '.nii.gz')
            label_path = os.path.join(datapath, dir_name, dir_name + img_suffix + '_' + contrast + disc_label_suffix + '.nii.gz')
            seg_path = os.path.join(datapath, dir_name, dir_name + img_suffix + '_' + contrast + seg_suffix + '.nii.gz' )
            if not os.path.exists(label_path):
                print(f'Error while importing {dir_name}\n {img_path} may not exist\n {label_path} may not exist, please check suffix {disc_label_suffix}\n')
            elif Image(label_path).change_orientation('RSP').data.shape==Image(img_path).change_orientation('RSP').data.shape:  # Shape not matching between image and labels
                print(f'Error with {dir_name}\n Image shape and label shape do not match')
            else:
                if os.path.exists(seg_path) and Image(seg_path).change_orientation('RSP').data.shape==Image(img_path).change_orientation('RSP').data.shape:  # Check if seg_shape == img_shape or create new seg:
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
    parser = argparse.ArgumentParser(description='Add ground truth coordinates to text file')

    ## Parameters
    # All mandatory                          
    parser.add_argument('--datapath', type=str, metavar='<folder>',
                        help='Path to data folder generated using src/bcm/utils/gather_data.py Example: ~/<your_dataset>/vertebral_data (Required)')                               
    parser.add_argument('-c', '--contrast', type=str, required=True,
                        help='MRI contrast: choices=["t1", "t2"] (Required)')
    parser.add_argument('-txt', '--out-txt-file', default='',
                        type=str, metavar='N',help='Generated txt file path (default="results/files/(datapath_basename)_(CONTRAST)_discs_coords.txt")')
    
    # All optional
    parser.add_argument('--suffix-img', type=str, default='',
                        help='Specify img suffix example: sub-250791(IMG_SUFFIX)_T2w.nii.gz (default= "")')
    parser.add_argument('--suffix-label-disc', type=str, default='_labels-disc-manual',
                        help='Specify label suffix example: sub-250791(IMG_SUFFIX)_T2w(DISC_LABEL_SUFFIX).nii.gz (default= "_labels-disc-manual")')
    parser.add_argument('--suffix-seg', type=str, default='_seg',
                        help='Specify segmentation label suffix example: sub-296085(IMG_SUFFIX)_T2w(SEG_SUFFIX).nii.gz (default= "_seg")')
    
    # Init output txt file if does not exist
    if not os.path.exists(parser.parse_args().out_txt_file):
        init_txt_file(parser.parse_args())

    # Run add_gt_coordinate_to_txt_file on input data
    add_gt_coordinate_to_txt_file(parser.parse_args())