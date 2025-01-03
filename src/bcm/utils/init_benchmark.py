import os
import argparse
import json
from pathlib import Path

from bcm.utils.utils import fetch_subject_and_session, fetch_contrast, fetch_bcm_paths


def init_txt_file(args, split='TESTING', init_discs=25):
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
    methods_str = 'subject_name contrast num_disc\n'
    txt_lines = [methods_str]
    
    # Create a dict to keep track of problematique redundant subject/contrasts associations
    subject_contrast_association = dict()
    duplicate_sub_cont = []
    if not os.path.exists(path_out):
        Path(path_out).parent.mkdir(parents=True, exist_ok=True)
        print("Creating TXT file:", path_out)

        # Load subject paths
        img_paths, _, _ = fetch_bcm_paths(config_data, split=split)
        
        # Initialize txt_file with subject_names and nb_discs_init
        print(f"Initializing txt file with subjects and {nb_discs_init} discs")
        for img_path in img_paths:
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
                    txt_lines += [sub_name + ' ' + contrast + ' ' + str(disc_num + 1) + '\n' for disc_num in range(nb_discs_init)]
            else:
                subject_contrast_association[sub_name] = [contrast]
                # construct subject lines line = subject_name contrast disc_num ground_truth_coord + methods...
                txt_lines += [sub_name + ' ' + contrast + ' ' + str(disc_num + 1) + '\n' for disc_num in range(nb_discs_init)]
        
        if duplicate_sub_cont:
            raise ValueError("Duplicate subject/contrast:\n" + '\n'.join(duplicate_sub_cont))

        with open(path_out,"w") as f:
            f.writelines(txt_lines)
        
        print(f"TXT file: {path_out} was created")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Init text file for benchmark')

    ## Parameters
    # All mandatory                          
    parser.add_argument('--config-data', type=str, metavar='<folder>', required=True,
                        help='Config JSON file where every label/image used for TESTING has its path specified ~/<your_path>/config_data.json (Required)')  
    parser.add_argument('-txt', '--out-txt-file', default='results/files/discs_coords.txt',
                        type=str, metavar='N',help='Generated txt file path (default: "results/files/discs_coords.txt")')
    
    # Init txt file
    init_txt_file(parser.parse_args())

    print('Benchmark initialization completed !')

