from __future__ import print_function, absolute_import
import os
import sys
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pickle
from torch.utils.data import DataLoader 
import cv2

parent_dir = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(parent_dir)

from utils.train_utils import image_Dataset
from utils.test_utils import CONTRAST

def create_skeleton(args):
    '''
    Create Skelet file to improve disc estimation of the hourglass network
    '''
    ## Load the training dataset
    trainset_path = args.datapath
    contrast = CONTRAST[args.contrast]
    with open(trainset_path, 'rb') as file_pi:
        full = pickle.load(file_pi)
    full[0] = full[0][:, :, :, :, 0]    

    ## Create a dataset loader
    full_dataset_train = image_Dataset(image_paths=full[0], target_paths=full[1], num_channel=args.ndiscs, use_flip = False)
    MRI_train_loader = DataLoader(full_dataset_train, batch_size= 1, shuffle=False, num_workers=0)

    All_skeletons = np.zeros((len(MRI_train_loader), 11, 2))
    Joint_counter = np.zeros((11, 1))
    for i, (input, target, vis) in enumerate(MRI_train_loader):
        target = target.numpy()
        mask = np.zeros((target.shape[2], target.shape[3]))
        for idc in range(target.shape[1]):
            mask += target[0, idc]
        mask = np.uint8(np.where(mask>0, 1, 0))
        mask = np.rot90(mask)
        num_labels, labels_im, states, centers = cv2.connectedComponentsWithStats(mask)
        centers = [t[::-1] for t in centers]
        skelet = np.zeros((11, 2))
        skelet[0:len(centers)-1] = centers[1:]
        Normjoint = np.linalg.norm(skelet[0]-skelet[4])
        for idx in range(1, len(centers)-1):
            skelet[idx] = (skelet[idx] - skelet[0]) / Normjoint

        skelet[0] *= 0
        
        All_skeletons[i] = skelet
        Joint_counter[0:len(centers)-1] += 1
        
    Skelet = np.sum(All_skeletons, axis= 0)   
    Skelet /= Joint_counter

    np.save(os.path.join(os.path.dirname(trainset_path), f'{contrast}_Skelet.npy'), Skelet)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training hourglass network')
    
    ## Parameters
    parser.add_argument('--datapath', type=str, required=True,
                        help='Path to trainset')
    parser.add_argument('-c', '--contrast', type=str, metavar='N', required=True,
                        help='MRI contrast')
    
    create_skeleton(parser.parse_args())  # Create skeleton file to improve hourglass accuracy during testing
