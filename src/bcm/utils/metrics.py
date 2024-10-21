#===================================================
## Authors: 
# - Lucas Rouhier ()
# - Reza Azad (rezazad68@gmail.com)
# - Nathan Molinier (nathan.molinier@gmail.com)
# - Morane Bienvenu
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
#===================================================

# Metrics computation
import numpy as np
#from sklearn.metrics import mean_squared_error

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
def compute_TP_and_FP(discs_gt, discs_pred):
    '''
    Compute true and false positive:
    
    missing_gt: numpy array of ground truth discs
    discs_pred: numpy array of the predicted discs
    
    return:
        TP: number of true positive detection
        FP: number of false positive detection
        false_pos_list: numpy array of false positive discs detections
    '''
    false_pos_list = []
    FP = 0
    TP = 0
    for disc in discs_pred:
        if disc not in discs_gt:
            FP += 1
            false_pos_list.append(disc)
        else:
            TP += 1
    return TP, FP, np.array(false_pos_list)

# compute False negative 
def compute_TN_and_FN(missing_gt, missing_pred):
    '''
    Compute true and false negative:
    
    missing_gt: numpy array with non present discs
    missing_pred: numpy array of the missed predictions
    
    return:
        TN: number of true negative detection
        FN: number of false negative detection
        false_neg_list: numpy array of false negative discs detections
    '''
    false_neg_list = []
    FN = 0
    TN = 0
    for disc in missing_pred:
        if disc not in missing_gt:
            FN += 1
            false_neg_list.append(disc)
        else:
           TN += 1 
    return TN, FN, np.array(false_neg_list)


#Compute MSE
def compute_MSE(pred,gt):
    #mse = mean_squared_error(pred_mask, gt_mask) #Les arguments d'entrée doivent être des tableaux
    '''
    Compute Mean Squared Error (MSE) between ground truth and predicted coordinates.
    
    gt: numpy array with the ground truth coords of the discs
    pred: numpy array with the prediction coords of the discs
    
    Returns: MSE value
    '''
    # Calculate L2 errors
    l2_errors = np.array([np.linalg.norm(gt[i] - pred[i]) for i in range(gt.shape[0])])
    
    # Calculate MSE
    mse = np.mean(l2_errors**2)
    return mse


def compute_dsc(gt_mask, pred_mask):
    """
    :param gt_mask: Ground truth mask used as the reference
    :param pred_mask: Prediction mask

    :return: dsc=2*intersection/(number of non zero pixels)
    """
    numerator = 2 * np.sum(gt_mask*pred_mask)
    denominator = np.sum(gt_mask) + np.sum(pred_mask)
    if denominator == 0:
        # Both ground truth and prediction are empty
        return 0
    else:
        return numerator / denominator   



