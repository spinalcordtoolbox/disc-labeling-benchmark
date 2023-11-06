import os
import json
import argparse
import shutil
import time
import glob
import torch
import cv2

from bcm.utils.utils import edit_subject_lines_txt_file, fetch_img_and_seg_paths, fetch_subject_and_session, fetch_contrast, tmp_create
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

        # Look for segmentation path
        add_subject = False
        back_up_seg_path = os.path.join(args.seg_folder, 'derivatives-seg', seg_path.split('derivatives/')[-1])
        if os.path.exists(seg_path) and Image(seg_path).change_orientation('RSP').data.shape==Image(img_path).change_orientation('RSP').data.shape:  # Check if seg_shape == img_shape or create new seg
            add_subject = True
        elif args.create_seg and os.path.exists(back_up_seg_path) and Image(back_up_seg_path).change_orientation('RSP').data.shape==Image(img_path).change_orientation('RSP').data.shape:
            seg_path = back_up_seg_path
            add_subject = True

        if subjectID=='sub-rennesMS062': #and add_subject: # A segmentation is available for projection
            # This inference is based on https://github.com/ivadomed/model_seg_sci/blob/main/packaging/run_inference_single_subject.py
            fname_file_out = back_up_seg_path.replace(f'{seg_suffix}.nii.gz', '_label-nnunet.nii.gz')  # path to the file with disc labels
            
            if os.path.exists(fname_file_out):
                # Extract discs coordinates
                discs_coords = extract_discs_coordinates(fname_file_out)
            else:
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

                # Copy .nii.gz file from tmpdir_nnunet to derivative folder with results
                os.makedirs(os.path.dirname(fname_file_out), exist_ok=True)
                pred_file = glob.glob(os.path.join(tmpdir_nnunet, '*.nii.gz'))[0]
                shutil.copyfile(pred_file, fname_file_out)

                print('Deleting the temporary folder...')
                # Delete the temporary folder
                shutil.rmtree(tmpdir)

                # Extract discs coordinates
                discs_coords = extract_discs_coordinates(fname_file_out)

            
            # Edit coordinates in txt file
            # line = subject_name contrast disc_num gt_coords sct_discs_coords hourglass_coords spinenet_coords
            split_lines = edit_subject_lines_txt_file(coords=discs_coords, txt_lines=split_lines, subject_name=sub_name, contrast=contrast, method_name='nnunet_coords')
        else:
            print(f'No segmentation is available for {img_path}')

    for num in range(len(split_lines)):
        split_lines[num] = ' '.join(split_lines[num])
        
    with open(txt_file,"w") as f:
        f.writelines(split_lines)

def extract_discs_coordinates(label_path):
    '''
    Extract discs coordinates from label_mask
    '''
    # Open label mask
    label_mask = Image(label_path).data

    num_labels, labels_im, states, centers = cv2.connectedComponentsWithStats(label_mask)

    return centers



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