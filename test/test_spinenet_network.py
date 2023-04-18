import sys
import os
import matplotlib.pyplot as plt
import argparse
from matplotlib.patches import Polygon, Circle 
import numpy as np
from spinalcordtoolbox.image import Image, get_dimension
from spinenet import SpineNet

parent_dir = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.insert(0, parent_dir)

from utils.test_utils import CONTRAST, VERT_DISC, swap_y_origin, coord2list, project_on_spinal_cord 

# sys.path.insert(0, '/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/code/disc-labeling-hourglass/utils')
# from test_utils import CONTRAST, VERT_DISC, swap_y_origin, coord2list, project_on_spinal_cord 

#---------------------------Test spinenet--------------------------
def test_spinenet(args, test_mode=False):
    '''
    Use spinenet to find the vertebrae discs coordinates and append them
    to a txt file
    '''
    
    datapath = os.path.abspath(args.sct_datapath)
    contrast = CONTRAST[args.contrast]
    txt_file = args.out_txt_file
    
    # load in spinenet
    spnt = SpineNet(device='cuda:0', verbose=True)
    
    # Extract txt file lines
    if not test_mode:
        prefix = 'sub'
        with open(txt_file,"r") as f:
            file_lines = f.readlines()
            split_lines = [line.split(' ') for line in file_lines]
    else:
        prefix = args.sub
        
    print('Processing with spinenet')
    for dir_name in os.listdir(datapath):
        if dir_name.startswith(prefix):
            subject_name = dir_name
            file_name = subject_name + '_' + contrast + '.nii.gz'
            img_path = os.path.join(datapath, dir_name, file_name)  # path to the original image
            
            # img_niftii --> 3D image: shape = (64, 320, 320)
            img_niftii = Image(img_path).change_orientation("RPI")
            nx, ny, nz, nt, px, py, pz, pt = get_dimension(img_niftii)
            nb_slice = 6 # Use less slices
            img = np.rot90(np.moveaxis(img_niftii.data, 0, -1)[:, :, nx//2-nb_slice//2:nx//2+nb_slice//2])
            
            # detect and identify vertebrae in scan. Note that pixel spacing information is required 
            # so SpineNet knows what size to split patches into.
            try:
                vert_dicts_niftii = spnt.detect_vb(img, px)
            except RuntimeError:
                print(f'Spinenet Fail detection on subject {subject_name}')
                vert_dicts_niftii = 'Fail'

            if vert_dicts_niftii != 'Fail':
                # extracting discs coordinates from vertebrae detections
                discs_coords = {}
                for vert_dict in vert_dicts_niftii:
                    vert = vert_dict['predicted_label']
                    poly_sum = np.zeros_like(vert_dict['polys'][0])
                    for idx in range(len(vert_dict['polys'])):
                        poly_sum += np.array(vert_dict['polys'][idx])
                    poly_mean = poly_sum/len(vert_dict['polys'])
                    
                    top_disc = VERT_DISC[vert]
                    if top_disc in discs_coords:
                        discs_coords[top_disc] = (discs_coords[top_disc] + poly_mean[-1,:])/2 # To improve the accuracy of the positioning, we calculate the average coordinate between the top and the bottom vertebrae
                    else:
                        discs_coords[top_disc] = poly_mean[-1,:] # Extract the coordinates of the top left corner which is the closest to the top disc (due to RPI orientation)
                    
                    bottom_disc = VERT_DISC[vert]+1
                    if bottom_disc in discs_coords:
                        discs_coords[bottom_disc] = (discs_coords[bottom_disc] + poly_mean[-2,:])/2 # To improve the accuracy of the positioning, we calculate the average coordinate between the top and the bottom vertebrae
                    else:
                        discs_coords[bottom_disc] = poly_mean[-2,:] # Extract the coordinates of the bottom left corner which is the closest to the bottom disc (due to RPI orientation)
                
                # Convert discs num and coords to numpy array
                discs_num, coords = np.array(list(discs_coords.keys())), np.array(list(discs_coords.values()))
                
                # Move y origin to the bottom of the image like Niftii convention
                coords = swap_y_origin(coords=coords, img_shape=img[:,:,0].shape)
                
                # Project on spinalcord for 2D comparison
                seg_path = os.path.join(datapath, subject_name, f'{subject_name}_{contrast}_seg.nii.gz' )
                coords = project_on_spinal_cord(coords=coords, seg_path=seg_path, disc_num=False, proj_2d=True)
                
                # Swap to coords convention [x, y] --> [lines(y), columns(x)]
                coords = coord2list(coords=coords)
                
            if not test_mode: 
                # Write coordinates in txt file
                # Edit txt_file --> line = subject_name contrast disc_num ground_truth_coord sct_label_vertebrae_coord hourglass_coord spinenet_coord
                subject_index = np.where((np.array(split_lines)[:,0] == subject_name) & (np.array(split_lines)[:,1] == contrast))  
                start_index = subject_index[0][0]  # Getting the first line for the subject in the txt file
                last_index = subject_index[0][-1]  # Getting the last line for the subject in the txt file
                max_ref_disc = int(split_lines[last_index][2])  # Getting the last refferenced disc num
                if vert_dicts_niftii != 'Fail': # Handle fail detection of spinenet
                    for disc_num, coord in zip(discs_num, coords):
                        coord_2D = '[' + str(coord[0]) + ',' + str(coord[1]) + ']\n'
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
                            new_line[6] = coord_2D
                            last_index += 1
                            split_lines.insert(last_index, new_line) # Add new disc detection to txt_file lines
                            max_ref_disc = disc_num
                        else:
                            split_lines[start_index + (disc_num-1)][6] = coord_2D
                else:
                    for i in range(last_index - start_index):
                        split_lines[start_index + i][6] = 'Fail'

    if not test_mode:
        for num in range(len(split_lines)):
            split_lines[num] = ' '.join(split_lines[num])
            
        with open(txt_file,"w") as f:
            f.writelines(split_lines)
    else:            
        return nb_slice, img, discs_coords, vert_dicts_niftii

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Extract discs coords from sct and hourglass')

    parser.add_argument('--sct-datapath', default="/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/preprocessed_data/vertebral_data", type=str,
                        help='SCT dataset path')                               
    parser.add_argument('-c', '--contrast', default='t2', type=str, metavar='N',
                        help='MRI contrast')
    parser.add_argument('-txt', '--out-txt-file', default= None,
                        type=str, metavar='N',help='Generated txt file')
    parser.add_argument('-sub', default= 'sub-juntendo750w06',
                        type=str, metavar='N',help='Generated txt file')
    
    nb_slice, img, discs_coords, vert_dicts_niftii = test_spinenet(parser.parse_args(), test_mode=True)
    fig = plt.figure(figsize=(40,40))
    for slice_idx in range(nb_slice):
        ax = fig.add_subplot(2,3,slice_idx+1)
        ax.imshow(img[:,:,slice_idx], cmap='gray')
        ax.set_title(f'Slice {slice_idx+1}', fontsize=60)
        ax.axis('off')
        for disc, coord in discs_coords.items():
            ax.add_patch(Circle(coord, radius=1, ec='r'))
            ax.text(coord[0]-15, coord[1], disc, color='r', fontsize=15)
        for vert_dict in vert_dicts_niftii:
            if slice_idx in vert_dict['slice_nos']:
                poly_idx = int(vert_dict['slice_nos'].index(slice_idx))
                poly = np.array(vert_dict['polys'][poly_idx])
                ax.add_patch(Polygon(poly, ec='y',fc='none'))
                ax.text(np.mean(poly[:,0]), np.mean(poly[:,1]), vert_dict['predicted_label'],c='y', ha='center',va='center', fontsize=15)

    fig.suptitle('Detected Vertebrae (all slices)', fontsize=100)
    fig.savefig('visualize/test_spinenet.png')