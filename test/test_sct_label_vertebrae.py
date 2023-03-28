import os
import sys
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.sys import run_proc

parent_dir = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(parent_dir)

from utils.test_utils import CONTRAST


#---------------------------Test Sct Label Vertebrae--------------------------
def test_sct_label_vertebrae(args):
    '''
    Use sct_deepseg_sc and sct_label_vertebrae to find the vertebrae discs coordinates and append them
    to a txt file
    '''
    datapath = os.path.abspath(args.sct_datapath)
    contrast = CONTRAST[args.contrast]
    
    # Get output file for discs extraction
    if args.out_txt_file is not None:
        txt_file = args.out_txt_file
    else:
        txt_file = os.path.join('files', f'{contrast}_hg{args.ndiscs}_discs_coords.txt')
        
    with open(txt_file,"r") as f:  # Checking already processed subjects from coords.txt
        file_lines = f.readlines()
        processed_subjects_with_contrast = [line.split(' ')[0] + '_' + line.split(' ')[1] for line in file_lines[1:]]  # Remove first line
        
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
            if (subject_name + '_' + contrast) not in processed_subjects_with_contrast:
                if discs_coords == 'Fail':  # SCT method error
                    lines = [subject_name + ' ' + contrast + ' ' + str(disc_num + 1) + ' ' + 'Fail' + ' ' + 'None' + ' ' + 'None' + '\n' for disc_num in range(11)] # To reorder the discs
                else:
                    lines = [subject_name + ' ' + contrast + ' ' + str(disc_num + 1) + ' ' + 'None' + ' ' + 'None' + ' ' + 'None' + '\n' for disc_num in range(11)] # To reorder the discs
                    last_referred_disc = 0
                    for coord in discs_coords:
                        coord_list = str(coord).split(',')
                        disc_num = int(float(coord_list[-1]))
                        coord_2d = '[' + str(coord_list[2]) + ',' + str(coord_list[1]) + ']'#  2D comparison of the models
                        if disc_num > 11:
                            print('More than 11 discs are visible')
                            print('Disc number', disc_num)
                            if disc_num == last_referred_disc + 1:  # Check if all the previous discs were also implemented
                                lines.append(subject_name + ' ' + contrast + ' ' + str(disc_num) + ' ' + coord_2d + ' ' + 'None' + ' ' + 'None' + '\n')
                                last_referred_disc = disc_num
                            else:
                                for i in range(disc_num - last_referred_disc - 1):
                                    lines.append(subject_name + ' ' + contrast + ' ' + str(last_referred_disc + 1 + i) + ' ' + 'None' + ' ' + 'None' + ' ' + 'None' + '\n')
                                lines.append(subject_name + ' ' + contrast + ' ' + str(disc_num) + ' ' + coord_2d + ' ' + 'None' + ' ' + 'None' + '\n')
                                last_referred_disc = disc_num
                        else:
                            lines[disc_num-1] = subject_name + ' ' + contrast + ' ' + str(disc_num) + ' ' + coord_2d + ' ' + 'None' + ' ' + 'None' + '\n'
                            last_referred_disc = disc_num
                with open(txt_file,"a") as f:
                    f.writelines(lines)