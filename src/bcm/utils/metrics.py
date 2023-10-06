#===================================================
## Authors: 
# - Lucas Rouhier ()
# - Reza Azad (rezazad68@gmail.com)
# - Nathan Molinier (nathan.molinier@gmail.com)
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
#===================================================

# Metrics computation
import numpy as np

from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import mean_squared_error

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





######## ajout ci-bas___ Morane


#Compute contour error using Euclidean distance
def compute_contour_error(seg_pred, seg_gt): #utilise seg_pred = imread(seg_path)? seg_gt=imread(seg_gt_path)?
    contour_error = np.mean(np.sqrt((seg_pred - seg_gt) ** 2))
    # Compute percentage error % (par rapport à la taille de l'image)
    height, width = seg_pred.shape
    total_pixels = height * width
    percentage_contour_error = (contour_error / total_pixels) * 100.0
#    contour_errors.append(percentage_contour_error)
#    print("Contour Error (%):", percentage_contour_error)
    return percentage_contour_error
    

#Compute MSE
def compute_MSE(pred_mask,gt_mask)
    "Calcule l'erreur quadratique moyenne entre la segmentation prédite et la vérité terrain"
    mse = mean_squared_error(pre_mask, gt_mask)
#    mse_scores.append(mse)
#    print("MSE:", mse)
    return mse

#Compute Hausdorff distance
def hausdorff_distance(set_pred, set_gt):
    """
    Calcule la distance de Hausdorff entre deux ensembles.

    Args:
    set_pred (numpy.ndarray): Le premier ensemble est un tableau 3D (masque binaire_pred)
    set_gt (numpy.ndarray): Le deuxième ensemble est un tableau 3D (masque binaire_gt).

    Returns:
    float: La distance de Hausdorff entre les deux ensembles.
    """
    # Trouver les coordonnées des points dans chaque ensemble
    points_set_pred = np.array(np.where(set_pred)).T
    points_set_gt = np.array(np.where(set_gt)).T

    # Calculer la distance de Hausdorff des deux ensembles
    hausdorff_pred_to_gt = directed_hausdorff(points_set_pred, points_set_gt)[0]
    hausdorff_gt_to_pred = directed_hausdorff(points_set_gt, points_set_pred)[0]

    # La distance de Hausdorff est la plus grande des deux distances
    hausdorff_distance = max(hausdorff_pred_to_gt, hausdorff_gt_to_pred)

    return hausdorff_distance
