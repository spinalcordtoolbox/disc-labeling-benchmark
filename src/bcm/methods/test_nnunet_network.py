import os
import json
import argparse
import shutil
import time
import glob
import torch
import numpy as np
import cc3d

from bcm.utils.utils import edit_subject_lines_txt_file, fetch_img_and_seg_paths, fetch_subject_and_session, fetch_contrast, tmp_create, project_on_spinal_cord, swap_y_origin
from bcm.utils.image import Image
from bcm.utils.config2parser import config2parser


from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
#from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

#---------------------------Test Sct Label Vertebrae--------------------------
def test_nnunet(args):
    '''
    Run nnunet inference for vertebral labeling and append the discs coordinates to a txt file
    '''
    txt_file = args.out_txt_file
    seg_suffix = args.suffix_seg

    # Get nnunet parameters
    config_nn = config2parser(args.config_nnunet)
    use_gpu = False
    
    # Read json file and create a dictionary
    with open(args.config_data, "r") as file:
        config_data = json.load(file)

    # Get image and segmentation paths
    img_paths, seg_paths = fetch_img_and_seg_paths(path_list=config_data['TESTING'], 
                                                   path_type=config_data['TYPE'], 
                                                   seg_suffix=seg_suffix, 
                                                   derivatives_path='derivatives/labels')
    
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

        # Look for segmentation path to project output coordinates
        add_subject = False
        back_up_seg_path = os.path.join(args.seg_folder, 'derivatives-seg', seg_path.split('derivatives/')[-1])
        if os.path.exists(seg_path) and Image(seg_path).change_orientation('RSP').data.shape==Image(img_path).change_orientation('RSP').data.shape:  # Check if seg_shape == img_shape or create new seg
            add_subject = True
        elif args.create_seg and os.path.exists(back_up_seg_path) and Image(back_up_seg_path).change_orientation('RSP').data.shape==Image(img_path).change_orientation('RSP').data.shape:
            seg_path = back_up_seg_path
            add_subject = True

        if add_subject: # A segmentation is available for projection
            # This inference is based on https://github.com/ivadomed/model_seg_sci/blob/main/packaging/run_inference_single_subject.py
            fname_file_out = back_up_seg_path.replace(f'{seg_suffix}.nii.gz', f'_label-nnunet-{str(config_nn.config_num)}.nii.gz')  # path to the file with disc labels
            
            if not os.path.exists(fname_file_out):
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

                # Create directory for nnUNet prediction
                tmpdir_nnunet = os.path.join(tmpdir, 'nnUNet_prediction')
                os.mkdir(tmpdir_nnunet)

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
                            output_folder_or_list_of_truncated_output_files=tmpdir_nnunet,
                            save_probabilities=False,
                            overwrite=True,
                            num_processes_preprocessing=3,
                            num_processes_segmentation_export=3
                )
                end = time.time()
                total_time = end - start

                # Copy .nii.gz file from tmpdir_nnunet to derivative folder with results to improve futur computation time
                os.makedirs(os.path.dirname(fname_file_out), exist_ok=True)
                pred_file = glob.glob(os.path.join(tmpdir_nnunet, '*.nii.gz'))[0]
                shutil.copyfile(pred_file, fname_file_out)

                print('Deleting the temporary folder...')
                # Delete the temporary folder
                shutil.rmtree(tmpdir)

            # Extract discs coordinates
            pred = Image(fname_file_out).change_orientation('RIP').data
            discs_coords = extract_discs_coordinates(pred)

            # Project coordinates onto the spinalcord
            proj_coords = project_on_spinal_cord(coords=discs_coords, seg_path=seg_path, disc_num=True, proj_2d=False)

            # Remove left-right coordinate
            proj_coords = proj_coords[:, 1:].astype(int)
            
            # Edit coordinates in txt file
            # line = subject_name contrast disc_num gt_coords sct_discs_coords hourglass_coords spinenet_coords
            split_lines = edit_subject_lines_txt_file(coords=proj_coords, txt_lines=split_lines, subject_name=sub_name, contrast=contrast, method_name='nnunet_coords')
        else:
            print(f'No segmentation is available for {img_path}')

    for num in range(len(split_lines)):
        split_lines[num] = ' '.join(split_lines[num])
        
    with open(txt_file,"w") as f:
        f.writelines(split_lines)


def extract_discs_coordinates(arr):
    '''
    Extract discs coordinates from arr
    :param arr: numpy array
    '''
    discs_coords = []
    if np.max(arr) > 1:
        for disc_num in range(1, np.max(arr)+1):
            discs_arr = np.where(arr==disc_num)
            if len(discs_arr[0]):
                discs_coords.append([int(discs_arr[0].mean()), int(discs_arr[1].mean()), int(discs_arr[2].mean()), disc_num])
    elif np.max(arr) == 1:
        centroids = cc3d.statistics(cc3d.connected_components(arr))['centroids'][1:] # Remove backgroud <0>
        centroids_sorted = centroids[np.argsort(-centroids[:,1])].astype(int) # Sort according to the vertical axis
        discs_coords = np.concatenate((centroids_sorted, np.expand_dims(np.arange(1, len(centroids_sorted)+1), axis=1)), axis=1) # Add discs numbers
    return discs_coords



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Add nnunet coordinates to text file')

    ## Parameters
    # All mandatory parameters                         
    parser.add_argument('--config-data', type=str, metavar='<folder>', required=True,
                        help='Config JSON file where every label/image used for TESTING has its path specified ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--config-nnunet', type=str, required=True,
                        help='Config file where nnunet training information are stored Example: Example: ~/<your_path>/config.json (Required)')  # nnunet config file
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
    
    # Run sct_label_vertebrae on input data
    test_nnunet(parser.parse_args())

    print('nnunet coordinates have been added')