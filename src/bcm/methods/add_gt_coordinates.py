import argparse
import numpy as np
import json

from bcm.utils.utils import project_on_spinal_cord, edit_subject_lines_txt_file, fetch_bcm_paths, fetch_subject_and_session, fetch_contrast
from bcm.utils.image import Image

def add_gt_coordinate_to_txt_file(args):
    '''
    Add ground truth coordinates to text file
    '''

    # Read json file and create a dictionary
    with open(args.config_data, "r") as file:
        config_data = json.load(file)
    
    # Check if labels are specified else don't compute ground truth
    if config_data['TYPE'] != 'LABEL-SEG':
        raise ValueError('Please use config with TYPE --> LABEL-SEG')
    else:
        txt_file = args.out_txt_file

        # Get image and segmentation paths
        img_paths, label_paths, seg_paths = fetch_bcm_paths(config_data)
        
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

            # Check if mismatch between images
            add_subject = False
            if Image(seg_path).change_orientation('RSP').data.shape==Image(label_path).change_orientation('RSP').data.shape:  # Check if seg_shape == label_shape
                add_subject = True
            
            if add_subject: # No mismatch detected
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
                print(f'Shape mismatch between {label_path} and {seg_path}')
        
        for num in range(len(split_lines)):
            split_lines[num] = ' '.join(split_lines[num])
            
        with open(txt_file,"w") as f:
            f.writelines(split_lines)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Add ground truth coordinates to text file')

    ## Parameters
    # All mandatory parameters                         
    parser.add_argument('--config-data', type=str, metavar='<folder>', required=True,
                        help='Config JSON file where every label/image used for TESTING has its path specified ~/<your_path>/config_data.json (Required)')                               
    parser.add_argument('-txt', '--out-txt-file', required=True,
                        type=str, metavar='N',help='Generated txt file path (e.g. "results/files/(CONTRAST)_discs_coords.txt") (Required)')
    
    # Run add_gt_coordinate_to_txt_file on input data
    add_gt_coordinate_to_txt_file(parser.parse_args())

    print('Ground truth coordinates have been added')