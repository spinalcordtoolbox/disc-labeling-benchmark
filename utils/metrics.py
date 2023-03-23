#===================================================
## Authors: 
# - Lucas Rouhier ()
# - Reza Azad (rezazad68@gmail.com)
# - Nathan Molinier (nathan.molinier@gmail.com)
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
#===================================================

# Metrics computation
import numpy as np


# looks for the closest points between real and predicted
def closest_node(node, nodes):
    nodes1 = np.asarray(nodes)
    dist_2 = np.sum((nodes1 - node) ** 2, axis=1)
    return np.argmin(dist_2)


# compute prediction L2-error with between predicted and closest real point.
def compute_L2_error(gt, pred):
    '''
    Compute L2 error:
    
    gt: numpy array with the ground truth coords of the discs
    pred: numpy array with the prediction coords of the discs
    
    out: L2_error for each coordinates
    '''
    return np.array([np.linalg.norm(gt[i] - pred[i]) for i in range(gt.shape[0])])


# compute error along the z axis between real and prediction with the closest node.
def compute_z_error(gt, pred):
    '''
    Compute vertical error:
    
    gt: numpy array with the ground truth coords of the discs
    pred: numpy array with the prediction coords of the discs
    
    out: Z_error for each coordinates
    '''
    return np.array([abs(gt[i,0] - pred[i,0]) for i in range(gt.shape[0])])


# compute False positive 
def false_pos(missing_gt, discs_pred):
    '''
    Compute false positive:
    
    missing_gt: numpy array of non present discs
    discs_pred: numpy array of the predicted discs
    
    return:
        c: number of false positive detection
        false_pos_list: numpy array of false positive discs detections
    '''
    false_pos_list = []
    c = 0
    for disc in discs_pred:
        if disc in missing_gt:
            c += 1
            false_pos_list.append(disc)
    return c, np.array(false_pos_list)

# compute False negative 
def false_neg(missing_gt, missing_pred):
    '''
    Compute false negative:
    
    missing_gt: numpy array with non present discs
    missing_pred: numpy array of the missed predictions
    
    return:
        c: number of false negative detection
        false_neg_list: numpy array of false negative discs detections
    '''
    false_neg_list = []
    c = 0
    for disc in missing_pred:
        if disc not in missing_gt:
            c += 1
            false_neg_list.append(disc)
    return c, np.array(false_neg_list)


