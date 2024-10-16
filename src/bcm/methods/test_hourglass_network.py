import os
import json
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

from bcm.utils.utils import CONTRAST, swap_y_origin, project_on_spinal_cord, edit_subject_lines_txt_file, fetch_bcm_paths, fetch_contrast, fetch_subject_and_session
from bcm.utils.image import Image
from bcm.utils.config2parser import config2parser
from bcm.utils.init_benchmark import init_txt_file

from dlh.models.hourglass import hg
from dlh.models.atthourglass import atthg
from dlh.utils.train_utils import image_Dataset, apply_preprocessing
from dlh.utils.data2array import get_midNifti
from dlh.utils.test_utils import extract_skeleton

#---------------------------Test Hourglass Network----------------------------
def test_hourglass(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_file = args.out_txt_file

    # Get hourglass training parameters
    config_hg = config2parser(args.config_hg)
    train_contrast = config_hg.train_contrast
    ndiscs = config_hg.ndiscs
    att = config_hg.att
    stacks = config_hg.stacks
    blocks = config_hg.blocks
    skeleton_dir = config_hg.skeleton_folder
    weights_dir = config_hg.weight_folder

    # Read json file and create a dictionary
    with open(args.config_data, "r") as file:
        config_data = json.load(file)
    
    # Fetch contrast info from config data
    data_contrast = CONTRAST[config_data['CONTRASTS']] # contrast_str is a unique string representing all the contrasts
    
    # Error if data contrast not in training
    for cont in data_contrast:
        if cont not in CONTRAST[train_contrast]:
            raise ValueError(f"Data contrast {cont} not used for training.")

    # Get image and segmentation paths
    img_paths, _, seg_paths = fetch_bcm_paths(config_data)

    # Load images
    print('loading images...')
    imgs_test, subjects_test, original_shapes = load_image_hg(img_paths=img_paths)

    # Load disc_coords txt file
    with open(txt_file,"r") as f:  # Checking already processed subjects from txt file
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]

    # Verify if skeleton exists before running test
    path_skeleton = os.path.join(skeleton_dir, f'{train_contrast}_Skelet_ndiscs_{ndiscs}.npy')
    if os.path.exists(path_skeleton):
        print(f'Processing with hourglass trained on contrast {train_contrast}')
        norm_mean_skeleton = np.load(path_skeleton)
        
        # Load network weights
        if att:
            model = atthg(num_stacks=stacks, num_blocks=blocks, num_classes=ndiscs)
        else:
            model = hg(num_stacks=stacks, num_blocks=blocks, num_classes=ndiscs)

        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(os.path.join(weights_dir, f'model_{train_contrast}_att_stacks_{stacks}_ndiscs_{ndiscs}'), map_location='cpu')['model_weights'])

        # Create Dataloader
        full_dataset_test = image_Dataset(images=imgs_test, 
                                        subjects_names=subjects_test,
                                        num_channel=ndiscs,
                                        use_flip = False,
                                        load_mode='test'
                                        ) 
        
        MRI_test_loader   = DataLoader(full_dataset_test, 
                                    batch_size= 1, 
                                    shuffle=False, 
                                    num_workers=0
                                    )
        model.eval()
        
        # Extract discs coordinates from the test set
        for i, (inputs, subject_name) in enumerate(MRI_test_loader):
            print(f'Running inference on {subject_name[0]}')

            # Project coordinate onto the spinal cord centerline
            print('Projecting labels onto the centerline')
            img_path = img_paths[i]
            seg_path = seg_paths[i]

            # Check if mismatch between images
            add_subject = False
            if Image(seg_path).change_orientation('RSP').data.shape==Image(img_path).change_orientation('RSP').data.shape:  # Check if seg_shape == img_shape
                add_subject = True
            
            if add_subject: # A segmentation is available for projection
                inputs = inputs.to(device)
                output = model(inputs) 
                output = output[-1]
                
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
                
                print('Extracting skeleton')
                try:
                    prediction, pred_discs_coords = extract_skeleton(inputs=inputs, outputs=output, norm_mean_skeleton=norm_mean_skeleton, Flag_save=True)
                    
                    # Convert pred_discs_coords to original image size
                    pred_shape = prediction[0,0].shape 
                    original_shape = original_shapes[i] 
                    pred = np.array([[(round(coord[0])/pred_shape[0])*original_shape[0], (round(coord[1])/pred_shape[1])*original_shape[1], int(disc_num)] for disc_num, coord in pred_discs_coords[0].items()]).astype(int)
                    
                    pred = project_on_spinal_cord(coords=pred, seg_path=seg_path, orientation='RSP', disc_num=True, proj_2d=True)
                    
                    # Swap axis prediction and ground truth
                    pred = swap_y_origin(coords=pred, img_shape=original_shape, y_pos=0).astype(int)  # Move y origin to the bottom of the image like Niftii convention
                
                except ValueError:
                    pred = np.array([]) # Fail

                # Edit coordinates in txt file
                # line = subject_name contrast disc_num gt_coords sct_discs_coords spinenet_coords hourglass_t1_coords hourglass_t2_coords hourglass_t1_t2_coords
                split_lines = edit_subject_lines_txt_file(coords=pred, txt_lines=split_lines, subject_name=sub_name, contrast=contrast, method_name=f'{args.method}_{train_contrast}')
            else:
                print(f'Shape mismatch between {img_path} and {seg_path}')
    else:
        raise ValueError(f'Path to skeleton {path_skeleton} does not exist'
                f'Please check if contrasts {train_contrast} was used for training')     
                
    for num in range(len(split_lines)):
        split_lines[num] = ' '.join(split_lines[num])
        
    with open(txt_file,"w") as f:
        f.writelines(split_lines)


def load_image_hg(img_paths):
    imgs = []
    subs = []
    shapes = []
    for path in img_paths:
        # Applying preprocessing steps
        image = apply_preprocessing(path)
        imgs.append(image)
        subject, sessionID, filename, contrast, echoID, acquisition = fetch_subject_and_session(path)
        subs.append(subject)
        shapes.append(get_midNifti(path).shape)
    return imgs, subs, shapes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add Hourglass Network coordinates to text file')

    ## Parameters
    # All mandatory parameters                         
    parser.add_argument('--config-data', type=str, metavar='<folder>', required=True,
                        help='Config JSON file where every label/image used for TESTING has its path specified ~/<your_path>/config_data.json (Required)')                               
    parser.add_argument('--config-hg', type=str, required=True,
                        help='Config file where hourglass training parameters are stored Example: Example: ~/<your_path>/config.json (Required)')  # Hourglass config file
    parser.add_argument('-txt', '--out-txt-file', default='results/files/discs_coords.txt',
                        type=str, metavar='N',help='Generated txt file path (default: "results/files/discs_coords.txt")')
    parser.add_argument('--method', default='hg',
                        type=str,help='Method name that will be added to the txt file (default="hg")')
    
    args = parser.parse_args()

    # Init txt file if it doesn't exist
    if not os.path.exists(args.out_txt_file):
        init_txt_file(args)

    # Run Hourglass Network on input data
    test_hourglass(args)

    print('Hourglass coordinates have been added')