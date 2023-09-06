import os
import json
import argparse
import numpy as np
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.sys import run_proc

from bcm.utils.utils import CONTRAST, SCT_CONTRAST, edit_subject_lines_txt_file, fetch_img_and_seg_paths, fetch_subject_and_session, fetch_contrast
from bcm.utils.init_txt_file import init_txt_file
from bcm.run.extract_discs_coords import parser_default


#---------------------------Test Sct Label Vertebrae--------------------------
def test_sct_label_vertebrae(args):
    '''
    Use sct_deepseg_sc and sct_label_vertebrae to find the vertebrae discs coordinates and append them
    to a txt file
    '''
    txt_file = args.out_txt_file
    seg_suffix = args.suffix_seg
    
    # Read json file and create a dictionary
    with open(args.config_data, "r") as file:
        config_data = json.load(file)

    # Get label paths
    label_paths = config_data['TESTING']
    
    # Get image and segmentation paths
    img_paths, seg_paths = fetch_img_and_seg_paths(path_list=config_data['TESTING'], 
                                                   path_type=config_data['TYPE'], 
                                                   seg_suffix=seg_suffix, 
                                                   derivatives_path='derivatives/labels')
    
    # Extract txt file lines
    with open(txt_file,"r") as f:
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
    
    print('Processing with sct_label_vertebrae')
    for img_path, label_path, seg_path in zip(img_paths, label_paths, seg_paths):
        # Create back up path for non provided segmentations
        back_up_seg_path = os.path.join(args.seg_folder, 'derivatives-seg', seg_path.split('derivatives')[-1])
        if os.path.exists(seg_path) and Image(seg_path).change_orientation('RSP').data.shape==Image(img_path).change_orientation('RSP').data.shape:  # Check if seg_shape == img_shape or create new seg
            status = 0
        elif os.path.exists(back_up_seg_path) and Image(back_up_seg_path).change_orientation('RSP').data.shape==Image(img_path).change_orientation('RSP').data.shape:
            status = 0
            seg_path = back_up_seg_path
        else:
            subjectID, _, _, _ = fetch_subject_and_session(img_path)
            contrast = fetch_contrast(img_path)

            # Create a new folder
            os.makedirs(os.path.dirname(back_up_seg_path), exist_ok=True)

            # Create a new segmentation file
            status, _ = run_proc(['sct_deepseg_sc',
                                    '-i', img_path, 
                                    '-c', SCT_CONTRAST[contrast],
                                    '-o', back_up_seg_path])
            seg_path = back_up_seg_path
        if status != 0:
            print(f'Fail segmentation for subject {subjectID}')
            discs_coords = np.array([]) # Fail
        else:
            disc_file_path = back_up_seg_path.replace('.nii.gz', '_labeled_discs.nii.gz')  # path to the file with disc labels
            if os.path.exists(disc_file_path):
                # retrieve all disc coords
                discs_coords = np.array([list(coord) for coord in Image(disc_file_path).change_orientation("RIP").getNonZeroCoordinates(sorting='value')]).astype(int)
                # keep only 2D coordinates
                discs_coords = discs_coords[:, 1:]
            else:
                status, _ = run_proc(['sct_label_vertebrae',
                                        '-i', img_path,
                                        '-s', seg_path,
                                        '-c', SCT_CONTRAST[contrast],
                                        '-ofolder', os.path.dirname(disc_file_path)], raise_exception=False)
                if status == 0:
                    discs_coords = np.array([list(coord) for coord in Image(disc_file_path).change_orientation("RIP").getNonZeroCoordinates(sorting='value')]).astype(int)
                    # keep only 2D coordinates
                    discs_coords = discs_coords[:, 1:]         
                else:
                    print(f'Fail sct_label_vertebrae for subject {subjectID}')
                    discs_coords = np.array([]) # Fail
        
        # Edit coordinates in txt file
        # line = subject_name contrast disc_num gt_coords sct_discs_coords hourglass_coords spinenet_coords
        split_lines = edit_subject_lines_txt_file(coords=discs_coords, txt_lines=split_lines, subject_name=subjectID, contrast=contrast, method_name='sct_discs_coords')

    for num in range(len(split_lines)):
        split_lines[num] = ' '.join(split_lines[num])
        
    with open(txt_file,"w") as f:
        f.writelines(split_lines)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Add sct_label_vertebrae coordinates to text file')

    ## Parameters
    # All mandatory parameters                         
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
    
    args = parser_default(parser.parse_args()) # Set default value for out-txt-file

    # Init output txt file if does not exist
    if not os.path.exists(args.out_txt_file):
        init_txt_file(args, split='TESTING', init_discs=11)

    # Run sct_label_vertebrae on input data
    test_sct_label_vertebrae(args)