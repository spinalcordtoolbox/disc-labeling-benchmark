import argparse
import json
import os
from pathlib import Path
import numpy as np
import subprocess

from bcm.utils.utils import project_on_spinal_cord, edit_subject_lines_txt_file, fetch_bcm_paths, fetch_subject_and_session, fetch_contrast, tmp_create, rmtree
from bcm.utils.cp2dir import cp2dir_mp
from bcm.utils.init_benchmark import init_txt_file
from bcm.utils.image import Image

def test_totalspineseg(args):
    """
    Use totalspineseg segmentation to extract discs and vertebrae labels
    """
    txt_file = args.out_txt_file
    
    # Read json file and create a dictionary
    with open(args.config_data, "r") as file:
        config_data = json.load(file)

    # Get image and segmentation paths
    img_paths, _, seg_paths = fetch_bcm_paths(config_data)
    
    # Load disc_coords txt file
    with open(txt_file,"r") as f:
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]

    # Copy files to a temporary folder
    tmp_dir = Path(tmp_create(basename='totalspineseg'))
    raw_dir = tmp_dir / 'raw'
    out_dir = tmp_dir / 'out'
    raw_dir.parent.mkdir(parents=True, exist_ok=True)

    # Create output directory totalspineseg
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # Copy input images to raw folder
    print('Copying input images to raw folder')
    cp2dir_mp(img_paths, raw_dir)

    # Run totalspineseg step 1
    subprocess.check_call([
        "totalspineseg", "--step1",
        str(raw_dir), str(out_dir)
    ])

    print('Processing with totalspineseg')
    level_dir = out_dir / 'step1_levels'
    for img_path, seg_path in zip(img_paths, seg_paths):
        img_path = Path(img_path)
        seg_path = Path(seg_path)

        # Get prediction path in level directory
        pred_path = level_dir / img_path.name

        # Check if mismatch between images
        add_subject = False
        if Image(str(seg_path)).change_orientation('RSP').data.shape==Image(str(img_path)).change_orientation('RSP').data.shape and Image(str(pred_path)).change_orientation('RSP').data.shape==Image(str(img_path)).change_orientation('RSP').data.shape:  # Check if seg_shape == img_shape
            add_subject = True
        
        if add_subject:
            # Fetch contrast, subject, session and echo
            subjectID, sessionID, _, _, echoID, acq = fetch_subject_and_session(str(img_path))
            sub_name = subjectID
            if acq:
                sub_name += f'_{acq}'
            if sessionID:
                sub_name += f'_{sessionID}'
            if echoID:
                sub_name += f'_{echoID}'
            contrast = fetch_contrast(str(img_path))

            # Extract discs coordinates
            pred_coords = np.array([list(coord) for coord in Image(str(pred_path)).change_orientation("RIP").getNonZeroCoordinates(sorting='value')]).astype(int)

            # Project on spinalcord
            pred_coords = project_on_spinal_cord(coords=pred_coords, seg_path=str(seg_path), orientation='RIP', disc_num=True, proj_2d=False)
            
            # Remove left-right coordinate
            pred_coords = pred_coords[:, 1:].astype(int)

            # Edit coordinates in txt file
            # line = subject_name contrast disc_num
            split_lines = edit_subject_lines_txt_file(coords=pred_coords, txt_lines=split_lines, subject_name=sub_name, contrast=contrast, method_name=args.method)
        else:
            print(f'Shape mismatch between {str(img_path)} and {str(seg_path)}')

    # Write txt file lines
    for num in range(len(split_lines)):
        split_lines[num] = ' '.join(split_lines[num])
        
    with open(txt_file,"w") as f:
        f.writelines(split_lines)
    
    # Remove tmp folder
    rmtree(tmp_dir)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Add totalspineseg coordinates to text file')

    ## Parameters
    # All mandatory parameters                         
    parser.add_argument('--config-data', type=str, metavar='<config>', required=True,
                        help='Config JSON file where every label/image used for TESTING has its path specified ~/<your_path>/config_data.json (Required)')
    parser.add_argument('-txt', '--out-txt-file', default='results/files/discs_coords.txt',
                        type=str, metavar='N',help='Generated txt file path (default: "results/files/discs_coords.txt")')
    parser.add_argument('--method', default='totalspineseg_coords',
                        type=str,help='Method name that will be added to the txt file (default="totalspineseg_coords")')
    
    args = parser.parse_args()

    # Init txt file if it doesn't exist
    if not os.path.exists(args.out_txt_file):
        init_txt_file(args)
    
    # Run sct_label_vertebrae on input data
    test_totalspineseg(parser.parse_args())

    print('totalspineseg coordinates have been added')