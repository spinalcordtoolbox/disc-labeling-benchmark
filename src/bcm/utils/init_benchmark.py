import os
import argparse
import json
import subprocess

from bcm.utils.utils import SCT_CONTRAST, get_img_path_from_label_path, fetch_subject_and_session, fetch_contrast, fetch_img_and_seg_paths
from bcm.utils.image import Image


def init_txt_file(args, split='TESTING', init_discs=11):
    """
    Initialize txt file where discs coordinates will be stored.

    :param out_txt_file: Generated txt file path (default="results/files/(CONTRAST)_discs_coords.txt")
    :param config_data_path: Config JSON file where every label/image used for TESTING has its path specified ~/<your_path>/config_data.json (Required)
    :param split: Split of the data used
    :param init_discs: Number of discs used to initialize the txt file
    """
    # Read json file and create a dictionary
    with open(args.config_data, "r") as file:
        config_data = json.load(file)
    
    # Initialize txt file with a constant number of line for each subject
    nb_discs_init = init_discs

    # Get outpath
    path_out = args.out_txt_file
    
    # Initialize txt file lines with method list
    methods_str = 'subject_name contrast num_disc gt_coords sct_discs_coords spinenet_coords hourglass_t1_coords hourglass_t2_coords hourglass_t1_t2_coords\n'
    txt_lines = [methods_str]
    
    # Create a dict to keep track of problematique redundant subject/contrasts associations
    subject_contrast_association = dict()
    duplicate_sub_cont = []
    if not os.path.exists(path_out):
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        print("Creating TXT file:", path_out)
        
        # Initialize txt_file with subject_names and nb_discs_init
        print(f"Initializing txt file with subjects and {nb_discs_init} discs")
        for path in config_data[split]:
            if config_data['TYPE'] == 'LABEL':
                img_path = get_img_path_from_label_path(path)
            if config_data['TYPE'] == 'IMAGE':
                img_path = path
            
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

            if sub_name in subject_contrast_association.keys():
                if contrast in subject_contrast_association[sub_name]:
                    duplicate_sub_cont.append(f'Duplicate {sub_name} with contrast {contrast}')
                else:
                    subject_contrast_association[sub_name].append(contrast)
                    # construct subject lines line = subject_name contrast disc_num ground_truth_coord + methods...
                    txt_lines += [sub_name + ' ' + contrast + ' ' + str(disc_num + 1) + ' None'*(len(methods_str.split(' '))-3) + '\n' for disc_num in range(nb_discs_init)]
            else:
                subject_contrast_association[sub_name] = [contrast]
                # construct subject lines line = subject_name contrast disc_num ground_truth_coord + methods...
                txt_lines += [sub_name + ' ' + contrast + ' ' + str(disc_num + 1) + ' None'*(len(methods_str.split(' '))-3) + '\n' for disc_num in range(nb_discs_init)]
        
        if duplicate_sub_cont:
            raise ValueError("Duplicate subject/contrast:\n" + '\n'.join(duplicate_sub_cont))

        with open(path_out,"w") as f:
            f.writelines(txt_lines)
        
        print(f"TXT file: {path_out} was created")


def init_sc_segmentation(args, split='TESTING'):
    # Read json file and create a dictionary
    with open(args.config_data, "r") as file:
        config_data = json.load(file)

    seg_suffix = args.suffix_seg

    # Get image and segmentation paths
    img_paths, seg_paths = fetch_img_and_seg_paths(path_list=config_data[split], 
                                                    path_type=config_data['TYPE'],
                                                    seg_suffix=seg_suffix
                                                    )
    
    print('Initializing SC segmentation for projection')
    for img_path, seg_path in zip(img_paths, seg_paths):
        contrast = fetch_contrast(img_path)

        # Create back up path for non provided segmentations
        back_up_seg_path = os.path.join(args.seg_folder, 'derivatives-seg', seg_path.split('derivatives/')[-1])
        if os.path.exists(seg_path) and Image(seg_path).change_orientation('RSP').data.shape==Image(img_path).change_orientation('RSP').data.shape:  # Check if seg_shape == img_shape or create new seg
            pass
        elif os.path.exists(back_up_seg_path) and Image(back_up_seg_path).change_orientation('RSP').data.shape==Image(img_path).change_orientation('RSP').data.shape:
            seg_path = back_up_seg_path
        else:
            print(f'{seg_path} does not exists, creating backup segmentation')
            # Create a new folder
            os.makedirs(os.path.dirname(back_up_seg_path), exist_ok=True)

            # Create a new segmentation file
            subprocess.check_call(['sct_deepseg_sc',
                                '-i', img_path, 
                                '-c', SCT_CONTRAST[contrast],
                                '-o', back_up_seg_path])
            seg_path = back_up_seg_path


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Init text file for benchmark')

    ## Parameters
    # All mandatory                          
    parser.add_argument('--config-data', type=str, metavar='<folder>', required=True,
                        help='Config JSON file where every label/image used for TESTING has its path specified ~/<your_path>/config_data.json (Required)')  
    parser.add_argument('-txt', '--out-txt-file', default='results/files/test_discs_coords.txt',
                        type=str, metavar='N',help='Generated txt file path (default="results/files/(CONTRAST)_discs_coords.txt")')
    
    # All methods
    parser.add_argument('--suffix-seg', type=str, default='_seg-manual',
                        help='Specify segmentation label suffix example: sub-296085_T2w(SEG_SUFFIX).nii.gz (default= "_seg")')
    parser.add_argument('--seg-folder', type=str, default='results',
                        help='Path to segmentation folder where non existing segmentations will be created. ' 
                        'These segmentations will be used to project labels onto the spinalcord (default="results")')
    parser.add_argument('--create-seg', type=bool, default=False,
                        help='To perform this benchmark, SC segmentation are needed for projection to compare the methods. '
                        'Set this variable to True to create segmentation using sct_deepseg_sc when not available')
    
    # Init txt file
    init_txt_file(parser.parse_args())

    # Init SC segmentation for projection
    if parser.parse_args().create_seg:
        init_sc_segmentation(parser.parse_args())

    print('Benchmark initialization completed !')

