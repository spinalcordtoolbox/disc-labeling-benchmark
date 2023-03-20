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
def mesure_err_disc(gt, pred):
    loss = []
    dis = []
    for i in range(len(gt[0])):
        node = np.array([gt[0][i], gt[1][i]])
        h = closest_node(node, pred)
        dis.append(np.linalg.norm(node - pred[h]))
    return dis

def compute_L2_error(gt, pred):
    '''
    gt: numpy array with the ground truth coords of the discs
    pred: numpy array with the prediction coords of the discs
    
    out: L2_error for each coordinate or:
        -1 if not coordinate for gt
        -2 if prediction fails
    '''
    L2_list = []
    for i in range(gt.shape[0]):
        if gt[i,0] != -1:
            if pred[i,0] != -1:
                L2_list.append(np.linalg.norm(gt[i] - pred[i]))
            else:
                L2_list.append(-2)
        else:
            L2_list.append(-1)
    return np.array(L2_list)


# compute error along the z axis between real and prediction with the closest node.
def mesure_err_z(gt, pred):
    z = []
    for i in range(len(gt[0])):
        node = np.array([gt[0][i], gt[1][i]])
        h = closest_node(node, pred)
        z.append(abs(node[0] - pred[h][0]))
    return z


# compute False positive by looking at points further than 10 mm from any points or groups of points attributed to same GT points
def Faux_pos(gt, pred):
    c = 0
    
    gt = np.transpose(gt)
    tot =len(gt)
    already_used = []
    for i in range(len(pred)):
        node = np.array([pred[i][0], pred[i][1]])
        h = closest_node(node, gt)
        #print(gt)
        if (abs(node[0] - gt[h][0])) > 10:
            c = c + 1

        elif h in already_used:

            #print(gt[h])
            #print('already_used')
            c = c + 1
    return c, tot


# compute false negative by looking at ground truth point further than 5mm than any predicted point
def Faux_neg(gt, pred):
    c = 0
    gt = np.transpose(gt)
    for i in range(len(gt)):
        node = np.array([gt[i][0], gt[i][1]])
        h = closest_node(node, pred)
        if (abs(node[0] - pred[h][0])) > 7:
            print(abs(node[0] - pred[h][0]))
            c = c + 1
    return c
