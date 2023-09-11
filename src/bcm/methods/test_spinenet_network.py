import os
import json
import matplotlib.pyplot as plt
import argparse
from matplotlib.patches import Polygon, Circle 
import numpy as np
from scipy.ndimage import zoom

from spinenet import SpineNet, io

from bcm.utils.utils import VERT_DISC, swap_y_origin, coord2list, project_on_spinal_cord, edit_subject_lines_txt_file, fetch_img_and_seg_paths, fetch_contrast, fetch_subject_and_session
from bcm.utils.image import Image, get_dimension

#---------------------------Test spinenet--------------------------
def test_spinenet(args):
    '''
    Use spinenet to find the vertebrae discs coordinates and append them
    to a txt file
    '''
    seg_suffix = args.suffix_seg
    txt_file = args.out_txt_file

    # load in spinenet
    spnt = SpineNet(device='cuda:0', verbose=True, scan_type='whole')
    
    # Read json file and create a dictionary
    with open(args.config_data, "r") as file:
        config_data = json.load(file)

    # Get image and segmentation paths
    img_paths, seg_paths = fetch_img_and_seg_paths(path_list=config_data['TESTING'], 
                                                   path_type=config_data['TYPE'],
                                                   seg_suffix=seg_suffix,
                                                   derivatives_path='derivatives/labels'
                                                   )

    with open(txt_file,"r") as f:
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
        
    print('Processing with spinenet')
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

        if add_subject: # A segmentation is available for projection
            # img_niftii --> 3D image: shape = (64, 320, 320)
            img_niftii = Image(img_path).change_orientation("RSP")
            nx, ny, nz, nt, px, py, pz, pt = get_dimension(img_niftii)
            pixel_spacing = np.array([py, pz])
            nb_slice = 12 # Use less slices
            if abs(px-py) < 0.2 or abs(px-pz) < 0.2: # True if isotropic according to right-left direction
                skip_slices = 4
                nx = nx//skip_slices
                slice_thickness = px*skip_slices
                #img = np.moveaxis(img_niftii.data, 0, -1)[:, :, nx//2-(nb_slice*skip_slices)//2:nx//2+(skip_slices*nb_slice)//2:skip_slices]
                img = np.moveaxis(zoom(img_niftii.data, (1/skip_slices, 1, 1)), 0, -1)[:, :, nx//2-nb_slice//2:nx//2+nb_slice//2] # Use zoom function to down sample the image to a more non-isotropic image
            else:
                slice_thickness = px
                img = np.moveaxis(img_niftii.data, 0, -1)[:, :, nx//2-nb_slice//2:nx//2+nb_slice//2]
            scan = io.SpinalScan(img, pixel_spacing, slice_thickness)
            #img = np.moveaxis(img_niftii.data, 0, -1)[:, :, nx//2-nb_slice//2:nx//2+nb_slice//2]

            # detect and identify vertebrae in scan. Note that pixel spacing information is required 
            # so SpineNet knows what size to split patches into.
            try:
                vert_dicts_niftii = spnt.detect_vb(img, scan.pixel_spacing[0])
            except RuntimeError:
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
                    
                    # Deal with coodinates outside of the image
                    poly_mean[:,0][poly_mean[:,0]>img.shape[1]-1] = img.shape[1]-1
                    poly_mean[:,1][poly_mean[:,1]>img.shape[0]-1] = img.shape[0]-1
                   
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
                discs_num, coords = np.transpose([np.array(list(discs_coords.keys()))]), np.array(list(discs_coords.values()))
                
                # Swap to coords convention [x, y] <--> [lines(y), columns(x)]
                coords = coord2list(coords=coords)
                
                # Concatenate discs num and coords
                coords = np.concatenate((coords, discs_num), axis=1)
                
                # Project on spinalcord for 2D comparison
                coords = project_on_spinal_cord(coords=coords, seg_path=seg_path, disc_num=True, proj_2d=True)
                
                # Move y origin to the bottom of the image like Niftii convention
                coords = swap_y_origin(coords=coords, img_shape=img[:,:,0].shape, y_pos=0).astype(int)

                # Write coordinates in txt file
                # line = subject_name contrast disc_num gt_coords sct_discs_coords hourglass_coords spinenet_coords
                split_lines = edit_subject_lines_txt_file(coords=coords, txt_lines=split_lines, subject_name=sub_name, contrast=contrast, method_name='spinenet_coords')
            else:
                coords = np.array([]) # Fail
                # Write coordinates in txt file
                # line = subject_name contrast disc_num gt_coords sct_discs_coords hourglass_coords spinenet_coords
                split_lines = edit_subject_lines_txt_file(coords=coords, txt_lines=split_lines, subject_name=sub_name, contrast=contrast, method_name='spinenet_coords')
        else:
            print(f'No segmentation is available for {img_path}')

    for num in range(len(split_lines)):
        split_lines[num] = ' '.join(split_lines[num])
        
    with open(txt_file,"w") as f:
        f.writelines(split_lines)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run test on spinenet')

    ## Parameters
    # All mandatory parameters                         
    parser.add_argument('--config-data', type=str, metavar='<folder>', required=True,
                        help='Config JSON file where every label/image used for TESTING has its path specified ~/<your_path>/config_data.json (Required)')                               
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
    
    # Run Hourglass Network on input data
    test_spinenet(parser.parse_args())

    print('Spinenet coordinates have been added')
    
    # if parser.parse_args().sub != '':
    #     nb_slice, img, discs_coords, vert_dicts_niftii = test_spinenet(parser.parse_args(), test_mode=True)
    #     fig = plt.figure(figsize=(40,40))
    #     for slice_idx in range(nb_slice):
    #         ax = fig.add_subplot(4,4,slice_idx+1)
    #         ax.imshow(img[:,:,slice_idx], cmap='gray')
    #         ax.set_title(f'Slice {slice_idx+1}', fontsize=60)
    #         ax.axis('off')
    #         for disc, coord in discs_coords.items():
    #             ax.add_patch(Circle(coord, radius=1, ec='r'))
    #             ax.text(coord[0]-15, coord[1], disc, color='r', fontsize=15)
    #         for vert_dict in vert_dicts_niftii:
    #             if slice_idx in vert_dict['slice_nos']:
    #                 poly_idx = int(vert_dict['slice_nos'].index(slice_idx))
    #                 poly = np.array(vert_dict['polys'][poly_idx])
    #                 ax.add_patch(Polygon(poly, ec='y',fc='none'))
    #                 ax.text(np.mean(poly[:,0]), np.mean(poly[:,1]), vert_dict['predicted_label'],c='y', ha='center',va='center', fontsize=15)

    #     fig.suptitle('Detected Vertebrae (all slices)', fontsize=100)
    #     fig.savefig('test/visualize/test_spinenet.png')
    # else: