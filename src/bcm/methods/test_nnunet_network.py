import os
import json
import argparse
import shutil
import time
import glob
import torch
import numpy as np
import cc3d

from bcm.utils.utils import edit_subject_lines_txt_file, fetch_bcm_paths, fetch_subject_and_session, fetch_contrast, tmp_create, project_on_spinal_cord
from bcm.utils.image import Image
from bcm.utils.config2parser import config2parser
from bcm.utils.init_benchmark import init_txt_file


from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
#from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

#---------------------------Test Sct Label Vertebrae--------------------------
def test_nnunet(args):
    '''
    Run nnunet inference for vertebral labeling and append the discs coordinates to a txt file
    '''
    txt_file = args.out_txt_file

    # Get nnunet parameters
    config_nn = config2parser(args.config_nnunet)
    use_gpu = False
    
    # Read json file and create a dictionary
    with open(args.config_data, "r") as file:
        config_data = json.load(file)

    # Get image and segmentation paths
    img_paths, _, seg_paths = fetch_bcm_paths(config_data)
    
    # Extract txt file lines
    with open(txt_file,"r") as f:
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
    
    print('Processing with nnunet')
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
            # This inference is based on https://github.com/ivadomed/model_seg_sci/blob/main/packaging/run_inference_single_subject.py
            
            # Create temporary directory in the temp to store the reoriented images
            tmpdir = tmp_create(basename='nnunet')            

            # Change the orientation to the same used for the training --> RSP and save the image
            fname_file_tmp = os.path.join(tmpdir, os.path.basename(img_path))
            Image(img_path).change_orientation('RSP').save(fname_file_tmp)

            # NOTE: for individual images, the _0000 suffix is not needed.
            # BUT, the images should be in a list of lists
            fname_file_tmp_list = [[fname_file_tmp]]

            # Use all the folds available in the model folder by default
            folds_avail = [int(f.split('_')[-1]) for f in os.listdir(config_nn.path_model) if f.startswith('fold_')]

            # Run nnUNet prediction
            print('Starting inference...')
            start = time.time()
            # directly call the object nnUNetPredictor
            predictor = nnUNetPredictor(
                        tile_step_size=config_nn.tile_step_size,     # changing it from 0.5 to 0.9 makes inference faster
                        use_gaussian=True,                      # applies gaussian noise and gaussian blur
                        use_mirroring=False,                    # test time augmentation by mirroring on all axes
                        perform_everything_on_gpu=True if use_gpu else False,
                        device=torch.device('cuda', 0) if use_gpu else torch.device('cpu'),
                        verbose=False,
                        verbose_preprocessing=False,
                        allow_tqdm=True
            )
            predictor.initialize_from_trained_model_folder(
                        model_training_output_dir=config_nn.path_model,
                        use_folds=folds_avail,
                        checkpoint_name='checkpoint_final.pth' if not config_nn.use_best_checkpoint else 'checkpoint_best.pth',
            )
            predictor.predict_from_files(
                        list_of_lists_or_source_folder=fname_file_tmp_list,
                        output_folder_or_list_of_truncated_output_files=tmpdir,
                        save_probabilities=False,
                        overwrite=True,
                        num_processes_preprocessing=3,
                        num_processes_segmentation_export=3
            )
            end = time.time()
            total_time = end - start

            # Copy .nii.gz file from tmpdir_nnunet to derivative folder with results to improve futur computation time
            fname_file_out = glob.glob(os.path.join(tmpdir, '*.nii.gz'))[0]

            # Extract discs coordinates
            pred = Image(fname_file_out).change_orientation('RIP').data
            if 'start_disc' in vars(config_nn).keys():
                discs_coords = extract_discs_coordinates(pred, start_idx=config_nn.start_disc)
            else:
                discs_coords = extract_discs_coordinates(pred)

            if discs_coords.size:
                # Project coordinates onto the spinalcord
                discs_coords = project_on_spinal_cord(coords=discs_coords, seg_path=seg_path, orientation='RIP', disc_num=True, proj_2d=False)

                if discs_coords.any():
                    # Remove left-right coordinate
                    discs_coords = discs_coords[:, 1:].astype(int)
            
            # Edit coordinates in txt file
            # line = subject_name contrast disc_num gt_coords sct_discs_coords hourglass_coords spinenet_coords
            split_lines = edit_subject_lines_txt_file(coords=discs_coords, txt_lines=split_lines, subject_name=sub_name, contrast=contrast, method_name=f'{args.method}_{str(config_nn.config_num)}')

            print('Deleting the temporary folder...')
            # Delete the temporary folder
            shutil.rmtree(tmpdir)
        else:
            print(f'Shape mismatch between {img_path} and {seg_path}')

    for num in range(len(split_lines)):
        split_lines[num] = ' '.join(split_lines[num])
        
    with open(txt_file,"w") as f:
        f.writelines(split_lines)


def extract_discs_coordinates(arr, start_idx=1):
    '''
    Extract discs coordinates from arr
    :param arr: numpy array
    '''
    if np.max(arr) > 1:
        discs_coords = []
        for disc_num in range(1, np.max(arr)+1):
            discs_arr = np.where(arr==disc_num)
            if len(discs_arr[0]):
                discs_coords.append([int(discs_arr[0].mean()), int(discs_arr[1].mean()), int(discs_arr[2].mean()), disc_num+start_idx-1])
        discs_coords = np.array(discs_coords)
    elif np.max(arr) == 1:
        centroids = cc3d.statistics(cc3d.connected_components(arr))['centroids'][1:] # Remove backgroud <0>
        centroids_sorted = centroids[np.argsort(-centroids[:,1])].astype(int) # Sort according to the vertical axis
        discs_coords = np.concatenate((centroids_sorted, np.expand_dims(np.arange(start_idx, len(centroids_sorted)+start_idx), axis=1)), axis=1) # Add discs numbers
    else:
        discs_coords = np.array([])
    return discs_coords



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Add nnunet coordinates to text file')

    ## Parameters
    # All mandatory parameters                         
    parser.add_argument('--config-data', type=str, metavar='<folder>', required=True,
                        help='Config JSON file where every label/image used for TESTING has its path specified ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--config-nnunet', type=str, required=True,
                        help='Config file where nnunet training information are stored Example: Example: ~/<your_path>/config.json (Required)')  # nnunet config file
    parser.add_argument('-txt', '--out-txt-file', default='results/files/discs_coords.txt',
                        type=str, metavar='N',help='Generated txt file path (default: "results/files/discs_coords.txt")')
    parser.add_argument('--method', default='nnunet',
                        type=str,help='Method name that will be added to the txt file (default="nnunet")')
    
    args = parser.parse_args()

    # Init txt file if it doesn't exist
    if not os.path.exists(args.out_txt_file):
        init_txt_file(args)

    # Run sct_label_vertebrae on input data
    test_nnunet(args)

    print('nnunet coordinates have been added')