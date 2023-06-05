#===================================================
## Authors: 
# - Lucas Rouhier ()
# - Reza Azad (rezazad68@gmail.com)
# - Nathan Molinier (nathan.molinier@gmail.com)
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
#===================================================

import os
import cv2
import numpy as np
import torch
from progress.bar import Bar
from sklearn.utils.extmath import cartesian
from torchvision.utils import make_grid

from bcm.utils.metrics import compute_L2_error, compute_z_error, compute_TP_and_FP, compute_TN_and_FN

from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.utils.fs import tmp_create, rmtree
from spinalcordtoolbox.utils.sys import run_proc


## Variables
CONTRAST = {'t1': ['T1w'],
            't2': ['T2w'],
            't2s':['T2star'],
            't1_t2': ['T1w', 'T2w']}

# Association between a vertebrae and the disc right above
VERT_DISC = {
    'C1':1,
    'C2':2,
    'C3':3,
    'C4':4,
    'C5':5,
    'C6':6,
    'C7':7,
    'T1':8,
    'T2':9,
    'T3':10,
    'T4':11,
    'T5':12,
    'T6':13,
    'T7':14,
    'T8':15,
    'T9':16,
    'T10':17,
    'T11':18,
    'T12':19,
    'L1':20,
    'L2':21,
    'L3':22,
    'L4':23,
    'L5':24,
    'S1':25
}


## Functions
def swap_y_origin(coords, img_shape, y_pos=1):
    '''
    This function returns a list of coords where the y origin coords was moved from top and bottom
    '''
    y_shape = img_shape[0]
    coords[:,y_pos] = y_shape - coords[:,y_pos]
    return coords

##
def save_discs_image(input_img, discs_images, out_path, target_th=0.5):
    clr_vis_Y = []
    hues = np.linspace(0, 179, discs_images.shape[0], dtype=np.uint8)
    blank_ch = 255*np.ones_like(discs_images[0], dtype=np.uint8)

    y_colored = np.zeros([discs_images.shape[1], discs_images.shape[2], 3], dtype=np.uint8)
    y_all = np.zeros([discs_images.shape[1], discs_images.shape[2]], dtype=np.uint8)
    
    for ych, hue_i in zip(discs_images, hues):
        ych = ych/np.max(np.max(ych))

        ych_hue = np.ones_like(ych, dtype=np.uint8)*hue_i
        ych = np.uint8(255*ych/np.max(ych))

        colored_ych = np.zeros_like(y_colored, dtype=np.uint8)
        colored_ych[:, :, 0] = ych_hue
        colored_ych[:, :, 1] = blank_ch
        colored_ych[:, :, 2] = ych
        colored_y = cv2.cvtColor(colored_ych, cv2.COLOR_HSV2BGR)

        y_colored += colored_y
        y_all += ych

    # Normalised image between [0,255] as integer
    x = (255*(input_img - np.min(input_img))/np.ptp(input_img)).astype(int)

    x_3ch = np.zeros([x.shape[0], x.shape[1], 3])
    for i in range(3):
        x_3ch[:, :, i] = x[:, :]

    img_mix = np.uint8(x_3ch*0.6 + y_colored*0.4)
    clr_vis_Y.append(img_mix)
            
    t = np.array(clr_vis_Y)
    t = np.transpose(t, [0, 3, 1, 2])
    trgts = make_grid(torch.Tensor(t), nrow=4)

    res = np.transpose(trgts.numpy(), (1,2,0))
    cv2.imwrite(out_path, res)

##
def visualize_discs(input_img, coords_list, out_path):
    coords_list = swap_y_origin(coords=coords_list, img_shape=input_img.shape, y_pos=0).tolist() # The y origin is at the top of the image
    discs_images = []
    for coord in coords_list:
        coord = [int(c) for c in coord]
        disc_img = np.zeros_like(input_img[:,:])
        disc_img[coord[0]-1:coord[0]+2, coord[1]-1:coord[1]+2] = [255, 255, 255]
        disc_img[:, :] = cv2.GaussianBlur(disc_img[:, :],(5,5),cv2.BORDER_DEFAULT)
        disc_img[:, :] = disc_img[:, :]/disc_img[:, :].max()*255
        discs_images.append(disc_img)
    discs_images = np.array(discs_images)
    save_discs_image(input_img, discs_images, out_path)

##
def coord2list(coords):
    '''
    This function swaps between coordinate referencial and list referencial
    [x, y] <--> [lines columns]
    '''
    return np.flip(coords,axis=1)

##
def str2array(coords):
    '''
    coords: numpy array of str coords
    
    returns numpy array of specified coordinates and None for not specified coordinates
    '''
    output_coords = []
    for coord in coords:
        if coord not in ['Fail', 'None', 'None\n']:
            coord_split = coord.split(',')
            output_coords.append([float(coord_split[0].split('[')[1]),float(coord_split[1].split(']')[0])])
        else:
            output_coords.append(None)
    return np.array(output_coords)

## 
def check_missing_discs(coords):
    '''
    coords: numpy array of coords
    return:
        output_coords: list of coords without missing discs
        missing_discs: list of missing discs detection
    '''
    output_coords = []
    missing_discs = []
    for num, coord in enumerate(coords):
        if coord is not None:
            output_coords.append(coord)
        else:
            missing_discs.append(num+1)
    return np.array(output_coords), np.array(missing_discs)
            
##
def project_on_spinal_cord(coords, seg_path, disc_num=True, proj_2d=False):
    '''
    This function projects discs coordinates onto the centerline using sct_label_utils
    
    coords: numpy array of coords
    return: numpy array of coords projected on the spinal cord
    '''
    # Get segmentation
    fname_seg = os.path.abspath(seg_path)
    seg = Image(fname_seg).change_orientation('RSP')
    
    # Create temp folder
    path_tmp = tmp_create(basename="label-vertebrae-project")
    
    if proj_2d:
        # Get middle slice of the image because last coordinate is missing to create a 3d object
        middle_slice = np.array([seg.data.shape[0]//2]*coords.shape[0])

    # Get transpose
    coords_t = np.rint(np.transpose(coords)).astype(int)
    
    # Create Image object of the referenced coordinates
    fname_coords = zeros_like(seg)
    
    if proj_2d:
        if disc_num:
            fname_coords.data[middle_slice, coords_t[0], coords_t[1]] = coords_t[2]
        else:
            fname_coords.data[middle_slice, coords_t[0], coords_t[1]] = 1
    else:
        if disc_num:
            fname_coords.data[coords_t[0], coords_t[1], coords_t[2]] = coords_t[3]
        else:
            fname_coords.data[coords_t[0], coords_t[1], coords_t[2]] = 1
    
    # Save referenced coords into temp folder
    discs_path = os.path.join(path_tmp, 'labels.nii.gz')
    fname_coords.save(path=discs_path, dtype='float32')
    
    # Compute projection
    out_path = os.path.join(path_tmp, 'proj_labels.nii.gz')
    status, _ = run_proc(['sct_label_utils',
                            '-i', fname_seg,
                            '-project-centerline', discs_path,
                            '-o', out_path], raise_exception=False)
    if status == 0:
        if disc_num:
            discs_coords = Image(out_path).getNonZeroCoordinates(sorting='value')
        else:
            discs_coords = Image(out_path).getNonZeroCoordinates(sorting='z', reverse_coord=True)
    else:
        print('Fail projection')
        
    if proj_2d:    
        if disc_num:
            output_coords = np.array([list(coord)[1:] for coord in discs_coords])
        else:
            output_coords = np.array([list(coord)[1:3] for coord in discs_coords])
    else:
        if disc_num:
            output_coords = np.array([list(coord) for coord in discs_coords])
        else:
            output_coords = np.array([list(coord)[:3] for coord in discs_coords])
    
    # Remove temporary folder
    rmtree(path_tmp)

    return output_coords

##
def edit_subject_lines_txt_file(coords, txt_lines, subject_name, contrast, method_name='hourglass_coord'):
    '''
    Write coordinates in txt file
    
    :param coords: 
        - numpy array of the 2D coordinates of discs plus discs num --> [[x1, y1, disc1], [x2, y2, disc2], ... [xN, yN, discN]]
        - if coords == np.array([]) # Fail --> method enable to perform on this subject
    :param txt_lines: list of the txt file lines
    :param subject_name: name of the subject in the text file
    :param contrast: contrast of the image
    :param method_name: name of the method being run where coordinates come from
    '''
    subject_index = np.where((np.array(txt_lines)[:,0] == subject_name) & (np.array(txt_lines)[:,1] == contrast))  
    start_index = subject_index[0][0]  # Getting the first line in the txt file
    last_index = subject_index[0][-1]  # Getting the last line for the subject in the txt file
    min_ref_disc = int(txt_lines[start_index][2])  # Getting the first refferenced disc num
    max_ref_disc = int(txt_lines[last_index][2])  # Getting the last refferenced disc num
    methods = txt_lines[0][:]
    methods[-1] = methods[-1].replace('\n','')
    method_idx = methods.index(method_name)
    nb_methods = len(methods) - 3 # The 3 first elements correspond to subject disc_num and contrast
    if method_idx == len(methods)-1:
        end_of_line = '\n'
    else:
        end_of_line = ''
        
    if coords.shape[0] == 0: # Fail
        print(f'{method_name} fails with subject {subject_name}')
        for i, _ in enumerate(range(min_ref_disc, max_ref_disc+1)):
            txt_lines[start_index + i][method_idx] = 'Fail' + end_of_line
    else:
        for i, disc_num in enumerate(range(min_ref_disc, max_ref_disc+1)):
            if disc_num in coords[:,-1]:
                idx = np.where(coords[:,-1] == disc_num)[0][0]
                txt_lines[start_index + i][method_idx] = '[' + str(coords[idx, 0]) + ',' + str(coords[idx, 1]) + ']' + end_of_line
            else:
                txt_lines[start_index + i][method_idx] = 'None' + end_of_line
        
        if max_ref_disc < np.max(coords[:,-1]):
            print(f'More discs found by {method_name}')
            for disc_num in coords[:,-1]:
                if disc_num > max_ref_disc:
                    print(f'Disc number {disc_num} was added')
                    new_line = [subject_name, contrast, str(disc_num)] + ['None']*(nb_methods-1) + ['None\n']
                    disc_shift = disc_num - max_ref_disc # Check if discs are missing between in the text file
                    if disc_shift != 1:
                        print(f'Adding {disc_shift-1} intermediate discs to txt file')
                        for shift in range(disc_shift-1):
                            last_index += 1
                            intermediate_line = new_line[:]
                            max_ref_disc += 1
                            intermediate_line[2] = str(max_ref_disc)
                            txt_lines.insert(last_index, intermediate_line) # Add intermediate lines to txt_file lines
                    idx = np.where(coords[:,-1] == disc_num)[0][0]
                    new_line[method_idx] = '[' + str(coords[idx, 0]) + ',' + str(coords[idx, 1]) + ']' + end_of_line
                    last_index += 1
                    txt_lines.insert(last_index, new_line) # Add new disc detection to txt_file lines
                    max_ref_disc = disc_num
    return txt_lines

##
def edit_metric_csv(result_dict, txt_lines, subject_name, contrast, method_name, nb_subjects):
    '''
    Calculate and edit csv file
    
    :param result_dict: result dictionary where all metrics will be gathered
    :param txt_lines: list of the input txt file lines
    :param subject_name: name of the subject in the text file
    :param contrast: contrast of the image used for testing
    :param method_name: name of the method on which metrics will be computed
    '''
    txt_lines = np.array(txt_lines)
    methods = txt_lines[0,:]
    methods[-1] = methods[-1].replace('\n','')
    method_idx = np.where(methods==method_name)[0][0]
    method_short = method_name.split('_coords')[0] # Remove '_coords' suffix
    
    subject_idx = np.where(methods=='subject_name')[0][0]
    discs_num_idx = np.where(methods=='num_disc')[0][0]
    contrast_idx = np.where(methods=='contrast')[0][0]
    gt_method_idx = np.where(methods=='gt_coords')[0][0]
    
    # Extract str coords and convert to numpy array, None stands for fail detections
    # TODO : Extract lines only corresponding to mentioned contrast
    discs_list = np.extract(txt_lines[:,subject_idx] == subject_name, txt_lines[:,discs_num_idx]).astype(int)
    gt_coords_list = str2array(np.extract(txt_lines[:,subject_idx] == subject_name, txt_lines[:, gt_method_idx]))
    pred_coords_list = str2array(np.extract(txt_lines[:,subject_idx] == subject_name, txt_lines[:,method_idx]))
    
    # Check for missing ground truth (only ground truth detections are considered as real discs)
    _, gt_missing_discs = check_missing_discs(gt_coords_list) # Numpy array of coordinates without missing detections + list of missing discs
        
    # Check for missing discs predictions
    pred_discs_list, pred_missing_discs = check_missing_discs(pred_coords_list) # Numpy array of coordinates without missing detections + list of missing discs
    
    # Calculate total prediction and true number of discs
    total_discs = discs_list.shape[0] - gt_missing_discs.shape[0]
    total_pred = discs_list.shape[0] - pred_missing_discs.shape[0]
    
    # Concatenate all the missing discs to compute metrics
    pred_and_gt_missing_discs = np.unique(np.append(pred_missing_discs, gt_missing_discs))
    pred_and_gt_missing_idx = np.in1d(discs_list, pred_and_gt_missing_discs) # get discs idx
    
    # Keep only coordinates that are present in both ground truth and prediction
    gt_coords_list = np.array(gt_coords_list[~pred_and_gt_missing_idx].tolist())
    pred_coords_list = np.array(pred_coords_list[~pred_and_gt_missing_idx].tolist())
    
    # Add subject to result dict
    if subject_name not in list(result_dict.keys()):
        result_dict[subject_name] = {}
    
    #-------------------------#
    # Compute L2 error
    #-------------------------#
    l2_pred = compute_L2_error(gt=gt_coords_list, pred=pred_coords_list)

    # Compute L2 error mean and std
    l2_pred_mean = np.mean(l2_pred) if l2_pred.size != 0 else 0
    l2_pred_std = np.std(l2_pred) if l2_pred.size != 0 else 0
    
    #--------------------------------#
    # Compute Z error
    #--------------------------------#
    z_err_pred = compute_z_error(gt=gt_coords_list, pred=pred_coords_list)
    
    # Compute z error mean and std
    z_err_pred_mean = np.mean(z_err_pred) if z_err_pred.size != 0 else 0
    z_err_pred_std = np.std(z_err_pred) if z_err_pred.size != 0 else 0

    #----------------------------------------------------#
    # Compute true and false positive rate (TPR and FPR)
    #----------------------------------------------------#
    gt_discs = discs_list[~np.in1d(discs_list, gt_missing_discs)]
    pred_discs = discs_list[~np.in1d(discs_list, pred_missing_discs)]

    TP_pred, FP_pred, FP_list_pred = compute_TP_and_FP(discs_gt=gt_discs, discs_pred=pred_discs)
    
    FPR_pred = FP_pred/total_pred if total_pred != 0 else 0
    TPR_pred = TP_pred/total_pred if total_pred != 0 else 0
    
    #----------------------------------------------------#
    # Compute true and false negative rate (TNR and FNR)
    #----------------------------------------------------#
    TN_pred, FN_pred, FN_list_pred = compute_TN_and_FN(missing_gt=gt_missing_discs, missing_pred=pred_missing_discs)
    
    FNR_pred = FN_pred/total_pred if total_pred != 0 else 1
    TNR_pred = TN_pred/total_pred if total_pred != 0 else 1
    
    #-------------------------------------------#
    # Compute dice score : DSC=2TP/(2TP+FP+FN)
    #-------------------------------------------#
    DSC_pred = 2*TP_pred/(2*TP_pred+FP_pred+FN_pred)
    
    ###################################
    # Add computed metrics to subject #
    ###################################
    
    # Add L2 error
    result_dict[subject_name][f'l2_mean_{method_short}'] = l2_pred_mean
    result_dict[subject_name][f'l2_std_{method_short}'] = l2_pred_std
    
    # Add Z error
    result_dict[subject_name][f'z_mean_{method_short}'] = z_err_pred_mean
    result_dict[subject_name][f'z_std_{method_short}'] = z_err_pred_std
    
    # Add true positive rate
    result_dict[subject_name][f'TPR_{method_short}'] = TPR_pred
    
    # Add false positive rate and FP list
    result_dict[subject_name][f'FP_list_{method_short}'] = FP_list_pred
    result_dict[subject_name][f'FPR_{method_short}'] = FPR_pred
    
    # Add true negative rate
    result_dict[subject_name][f'TNR_{method_short}'] = TNR_pred
    
    # Add false negative rate and FN list
    result_dict[subject_name][f'FN_list_{method_short}'] = FN_list_pred
    result_dict[subject_name][f'FNR_{method_short}'] = FNR_pred
    
    # Add dice score
    result_dict[subject_name][f'DSC_{method_short}'] = DSC_pred
    
    # Add total number of discs
    result_dict[subject_name]['tot_discs'] = total_discs
    result_dict[subject_name][f'tot_pred_{method_short}'] = total_pred
    
    ######################################
    # Add total mean of computed metrics #
    ######################################
    
    if 'total' not in list(result_dict.keys()):
        ## Init total dict for results
        result_dict['total'] = {}
    
    if f'l2_mean_{method_short}' not in list(result_dict['total'].keys()):
        # Init L2 error
        result_dict['total'][f'l2_mean_{method_short}'] = 0
        result_dict['total'][f'l2_std_{method_short}'] = 0
        
        # Init Z error
        result_dict['total'][f'z_mean_{method_short}'] = 0
        result_dict['total'][f'z_std_{method_short}'] = 0
        
        # Init true positive rate
        result_dict['total'][f'TPR_{method_short}'] = 0
        
        # Init false positive rate
        result_dict['total'][f'FPR_{method_short}'] = 0
        
        # Init true negative rate
        result_dict['total'][f'TNR_{method_short}'] = 0
        
        # Init false negative rate
        result_dict['total'][f'FNR_{method_short}'] = 0
        
        # Init dice score
        result_dict['total'][f'DSC_{method_short}'] = 0
    
    # Add L2 error
    result_dict['total'][f'l2_mean_{method_short}'] += l2_pred_mean/nb_subjects
    result_dict['total'][f'l2_std_{method_short}'] += l2_pred_std/nb_subjects
    
    # Add Z error
    result_dict['total'][f'z_mean_{method_short}'] += z_err_pred_mean/nb_subjects
    result_dict['total'][f'z_std_{method_short}'] += z_err_pred_std/nb_subjects
    
    # Add true positive rate
    result_dict['total'][f'TPR_{method_short}'] += TPR_pred/nb_subjects
    
    # Add false positive rate
    result_dict['total'][f'FPR_{method_short}'] += FPR_pred/nb_subjects
    
    # Add true negative rate
    result_dict['total'][f'TNR_{method_short}'] += TNR_pred/nb_subjects
    
    # Add false negative rate
    result_dict['total'][f'FNR_{method_short}'] += FNR_pred/nb_subjects
    
    # Add dice score
    result_dict['total'][f'DSC_{method_short}'] += DSC_pred/nb_subjects
    
    return result_dict, pred_discs_list