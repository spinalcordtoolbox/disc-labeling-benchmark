import os
import argparse
import numpy as np
import json

from bcm.utils.utils import swap_y_origin, project_on_spinal_cord, edit_subject_lines_txt_file, fetch_img_and_seg_paths, fetch_subject_and_session, fetch_contrast
from bcm.utils.image import Image

def add_gt_coordinate_to_txt_file(args):
    '''
    Add ground truth coordinates to text file
    '''

    # Read json file and create a dictionary
    with open(args.config_data, "r") as file:
        config_data = json.load(file)
    
    # Check if labels are specified else don't compute ground truth
    if config_data['TYPE'] == 'IMAGE':
        print("#####################################################################################################################################")
        print("#################### Paths to labels were not specified --> ground truth coordinates won't be added to the benchmark ####################")
        print("#####################################################################################################################################")
    elif config_data['TYPE'] == 'LABEL':
        txt_file = args.out_txt_file
        seg_suffix = args.suffix_seg

        # Get label paths
        label_paths = [os.path.join(config_data['DATASETS_PATH'], path) for path in config_data['TESTING']]

        # Get image and segmentation paths
        img_paths, seg_paths = fetch_img_and_seg_paths(path_list=config_data['TESTING'], 
                                                       path_type=config_data['TYPE'],
                                                       datasets_path=config_data['DATASETS_PATH'],
                                                       seg_suffix=seg_suffix,
                                                       derivatives_path='derivatives/labels')
        
        # Load disc_coords txt file
        with open(txt_file,"r") as f:  # Checking already processed subjects from txt file
            file_lines = f.readlines()
            split_lines = [line.split(' ') for line in file_lines]
        
        print('Adding ground truth coords')
        for img_path, label_path, seg_path in zip(img_paths, label_paths, seg_paths):
            
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

            # Look for segmentation path
            add_subject = False
            back_up_seg_path = os.path.join(args.seg_folder, 'derivatives-seg', seg_path.split('derivatives/')[-1])
            if os.path.exists(seg_path) and Image(seg_path).change_orientation('RSP').data.shape==Image(img_path).change_orientation('RSP').data.shape:  # Check if seg_shape == img_shape or create new seg
                add_subject = True
            elif args.create_seg and os.path.exists(back_up_seg_path) and Image(back_up_seg_path).change_orientation('RSP').data.shape==Image(img_path).change_orientation('RSP').data.shape:
                seg_path = back_up_seg_path
                add_subject = True
            
            if add_subject: # A segmentation is available for projection
                img_gt = Image(label_path).change_orientation('RIP')
                gt_coord = np.array([list(coord) for coord in img_gt.getNonZeroCoordinates(sorting='value') if coord[-1] < 25]).astype(int) # Remove labels superior to 25, especially 49 and 50 that correspond to the pontomedullary groove (49) and junction (50)
                
                # Project on spinalcord
                gt_coord = project_on_spinal_cord(coords=gt_coord, seg_path=seg_path, orientation='RIP', disc_num=True, proj_2d=False)
                
                # Remove thinkness coordinate
                gt_coord = gt_coord[:, 1:].astype(int)
                            
                # Edit coordinates in txt file
                # line = subject_name contrast disc_num gt_coords sct_discs_coords spinenet_coords hourglass_t1_coords hourglass_t2_coords hourglass_t1_t2_coords
                split_lines = edit_subject_lines_txt_file(coords=gt_coord, txt_lines=split_lines, subject_name=sub_name, contrast=contrast, method_name='gt_coords')
            else:
                print(f'No segmentation is available for {img_path}')
        
        for num in range(len(split_lines)):
            split_lines[num] = ' '.join(split_lines[num])
            
        with open(txt_file,"w") as f:
            f.writelines(split_lines)
    else:
        raise ValueError(f'Path TYPE {config_data["TYPE"]} is not defined')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Add ground truth coordinates to text file')

    ## Parameters
    # All mandatory parameters                         
    parser.add_argument('--config-data', type=str, metavar='<folder>', required=True,
                        help='Config JSON file where every label/image used for TESTING has its path specified ~/<your_path>/config_data.json (Required)')                               
    parser.add_argument('-txt', '--out-txt-file', required=True,
                        type=str, metavar='N',help='Generated txt file path (e.g. "results/files/(CONTRAST)_discs_coords.txt") (Required)')

    # All methods
    parser.add_argument('--suffix-seg', type=str, default='_seg-manual',
                        help='Specify segmentation label suffix example: sub-296085_T2w(SEG_SUFFIX).nii.gz (default= "_seg")')
    parser.add_argument('--seg-folder', type=str, default='results',
                        help='Path to segmentation folder where non existing segmentations will be created. ' 
                        'These segmentations will be used to project labels onto the spinalcord (default="results")')
    parser.add_argument('--create-seg', type=bool, default=False,
                        help='To perform this benchmark, SC segmentation are needed for projection to compare the methods. '
                        'Set this variable to True to create segmentation using sct_deepseg_sc when not available')
    
    # Run add_gt_coordinate_to_txt_file on input data
    add_gt_coordinate_to_txt_file(parser.parse_args())

    print('Ground truth coordinates have been added')