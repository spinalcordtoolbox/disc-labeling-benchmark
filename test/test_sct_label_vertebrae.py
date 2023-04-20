import os
import numpy as np
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.sys import run_proc

from dlh.utils.test_utils import CONTRAST


#---------------------------Test Sct Label Vertebrae--------------------------
def test_sct_label_vertebrae(args):
    '''
    Use sct_deepseg_sc and sct_label_vertebrae to find the vertebrae discs coordinates and append them
    to a txt file
    '''
    datapath = os.path.abspath(args.sct_datapath)
    contrast = CONTRAST[args.contrast]
    txt_file = args.out_txt_file
    
    # Extract txt file lines
    with open(txt_file,"r") as f:
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
    
    print('Processing with sct_label_vertebrae')
    for dir_name in os.listdir(datapath):
        if dir_name.startswith('sub'):
            file_name = dir_name + '_' + contrast + '.nii.gz'
            file_path = os.path.join(datapath, dir_name, file_name)  # path to the original image
            seg_path = file_path.replace('.nii.gz', '_seg.nii.gz')  # path to the spinal cord segmentation
            if os.path.exists(seg_path):
                pass
            else:
                status, _ = run_proc(['sct_deepseg_sc',
                                        '-i', file_path, 
                                        '-c', args.contrast,
                                        '-o', seg_path])
                if status != 0:
                    print('Fail segmentation')
                    discs_coords = 'Fail'
            
            disc_file_path = file_path.replace('.nii.gz', '_seg_labeled_discs.nii.gz')  # path to the file with disc labels
            if os.path.exists(disc_file_path):
                # retrieve all disc coords
                discs_coords = Image(disc_file_path).change_orientation("RPI").getNonZeroCoordinates(sorting='value')
            else:
                status, _ = run_proc(['sct_label_vertebrae',
                                            '-i', file_path,
                                            '-s', file_path.replace('.nii.gz', '_seg.nii.gz'),
                                            '-c', args.contrast,
                                            '-ofolder', os.path.join(datapath, dir_name)], raise_exception=False)
                if status == 0:
                    discs_coords = Image(disc_file_path).change_orientation("RPI").getNonZeroCoordinates(sorting='value')
                else:
                    print('Exit value 1')
                    print('Fail sct_label_vertebrae')
                    discs_coords = 'Fail'

            subject_name = dir_name
            subject_index = np.where((np.array(split_lines)[:,0] == subject_name) & (np.array(split_lines)[:,1] == contrast))  
            start_index = subject_index[0][0]  # Getting the first line for the subject in the txt file
            last_index = subject_index[0][-1]  # Getting the last line for the subject in the txt file
            max_ref_disc = int(split_lines[last_index][2])  # Getting the last refferenced disc num
            # Edit txt_file --> line = subject_name contrast disc_num ground_truth_coord sct_label_vertebrae_coord hourglass_coord spinenet_coord    
            if discs_coords == 'Fail':  # SCT method error
                for i in range(last_index - start_index):
                    split_lines[start_index + i][4] = 'Fail'
            else:
                for coord in discs_coords:
                    coord_list = list(coord)
                    disc_num = int(coord_list[-1])
                    coord_2d = '[' + str(coord_list[2]) + ',' + str(coord_list[1]) + ']' #  Because 2D comparison only of the models
                    if disc_num > max_ref_disc:
                        print('More discs found')
                        print('Disc number', disc_num)
                        new_line = [subject_name, contrast, str(disc_num), 'None', 'None', 'None', 'None\n']
                        disc_shift = disc_num - max_ref_disc # Check if discs are missing between in the text file
                        if disc_shift != 1:
                            print(f'Adding intermediate {disc_shift-1} discs to txt file')
                            for shift in range(disc_shift-1):
                                last_index += 1
                                intermediate_line = new_line[:]
                                max_ref_disc += 1
                                intermediate_line[2] = str(max_ref_disc)
                                split_lines.insert(last_index, intermediate_line) # Add intermediate lines to txt_file lines
                        new_line[4] = coord_2d
                        last_index += 1
                        split_lines.insert(last_index, new_line) # Add new disc detection to txt_file lines
                        max_ref_disc = disc_num
                    else:
                        split_lines[start_index + (disc_num-1)][4] = coord_2d
                    
    for num in range(len(split_lines)):
        split_lines[num] = ' '.join(split_lines[num])
        
    with open(txt_file,"w") as f:
        f.writelines(split_lines)