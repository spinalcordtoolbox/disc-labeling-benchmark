import os
import json
import argparse
import numpy as np
import subprocess

from bcm.utils.utils import SCT_CONTRAST, edit_subject_lines_txt_file, fetch_bcm_paths, fetch_subject_and_session, fetch_contrast, tmp_create, rmtree
from bcm.utils.image import Image


#---------------------------Test Sct Label Vertebrae--------------------------
def test_sct_label_vertebrae(args):
    '''
    Use sct_deepseg_sc and sct_label_vertebrae to find the vertebrae discs coordinates and append them
    to a txt file
    '''
    txt_file = args.out_txt_file
    
    # Read json file and create a dictionary
    with open(args.config_data, "r") as file:
        config_data = json.load(file)

    # Get image and segmentation paths
    img_paths, _, seg_paths = fetch_bcm_paths(config_data)
    
    # Extract txt file lines
    with open(txt_file,"r") as f:
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
    
    print('Processing with sct_label_vertebrae')
    for img_path, seg_path in zip(img_paths, seg_paths):

        # Fetch contrast, subject, session and echo
        subjectID, sessionID, _, _, echoID, acq = fetch_subject_and_session(img_path)
        sub_name = subjectID
        if acq:
            sub_name += f'_{acq}'
        if sessionID:
            sub_name += f'_{sessionID}'
        if echoID:
            sub_name += f'_{echoID}'
        contrast = fetch_contrast(img_path)

        # Check if mismatch between images
        add_subject = False
        if Image(seg_path).change_orientation('RSP').data.shape==Image(img_path).change_orientation('RSP').data.shape:  # Check if seg_shape == img_shape
            add_subject = True
        
        if add_subject: # A segmentation is available for projection
            tmp_dir = tmp_create(basename="sct-label-vertebrae")
            if os.path.exists(disc_file_path):
                # retrieve all disc coords
                discs_coords = np.array([list(coord) for coord in Image(disc_file_path).change_orientation("RIP").getNonZeroCoordinates(sorting='value')]).astype(int)
                # keep only 2D coordinates
                discs_coords = discs_coords[:, 1:]
            else:
                # Check if dirname exists
                if not os.path.exists(os.path.dirname(disc_file_path)):
                    os.makedirs(os.path.dirname(disc_file_path))

           
                out=subprocess.run(['sct_label_vertebrae',
                                        '-i', img_path,
                                        '-s', seg_path,
                                        '-c', SCT_CONTRAST[contrast],
                                        '-ofolder', tmp_dir])
                if out.returncode == 0:
                    disc_file_path = os.path.join(tmp_dir, img_path.replace('.nii.gz', '_seg_labeled_discs.nii.gz'))
                    discs_coords = np.array([list(coord) for coord in Image(disc_file_path).change_orientation("RIP").getNonZeroCoordinates(sorting='value')]).astype(int)
                    # keep only 2D coordinates
                    discs_coords = discs_coords[:, 1:]
                    rmtree(tmp_dir)      
                else:
                    print(f'Fail sct_label_vertebrae for subject {sub_name}')
                    discs_coords = np.array([]) # Fail
            
            # Edit coordinates in txt file
            # line = subject_name contrast disc_num gt_coords sct_discs_coords hourglass_coords spinenet_coords
            split_lines = edit_subject_lines_txt_file(coords=discs_coords, txt_lines=split_lines, subject_name=sub_name, contrast=contrast, method_name='sct_discs_coords')
        else:
            print(f'Shape mismatch between {img_path} and {seg_path}')

    for num in range(len(split_lines)):
        split_lines[num] = ' '.join(split_lines[num])
        
    with open(txt_file,"w") as f:
        f.writelines(split_lines)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Add sct_label_vertebrae coordinates to text file')

    ## Parameters
    # All mandatory parameters                         
    parser.add_argument('--config-data', type=str, metavar='<folder>', required=True,
                        help='Config JSON file where every label/image used for TESTING has its path specified ~/<your_path>/config_data.json (Required)')
    parser.add_argument('-txt', '--out-txt-file', required=True,
                        type=str, metavar='N',help='Generated txt file path (e.g. "results/files/(CONTRAST)_discs_coords.txt") (Required)')                             
    
    # Run sct_label_vertebrae on input data
    test_sct_label_vertebrae(parser.parse_args())

    print('sct_label_vertebrae coordinates have been added')