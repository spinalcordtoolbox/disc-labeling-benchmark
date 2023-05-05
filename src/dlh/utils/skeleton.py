from __future__ import print_function, absolute_import
import os
import sys
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pickle
from torch.utils.data import DataLoader 
import cv2

from dlh.utils.train_utils import image_Dataset
from dlh.utils.test_utils import CONTRAST, load_niftii_split

def create_skeleton(args):
    '''
    Create Skelet file to improve disc estimation of the hourglass network
    '''
    ## Load the training dataset
    datapath = args.datapath
    contrasts = CONTRAST[args.contrasts]
    ndiscs = args.ndiscs 
    out_dir = os.path.join(datapath,'skeletons')

    
    # Loading images for training
    print('loading images...')
    imgs_train, masks_train, discs_labels_train, subjects_train, _ = load_niftii_split(datapath=datapath, 
                                                                                   contrasts=contrasts, 
                                                                                   split='train', 
                                                                                   split_ratio=args.split_ratio)

    ## Create a dataset loader
    full_dataset_train = image_Dataset(images=imgs_train, 
                                       targets=masks_train,
                                       discs_labels_list=discs_labels_train,
                                       subjects_names=subjects_train,
                                       num_channel=args.ndiscs,
                                       use_flip = True,
                                       load_mode='val'
                                       )
    
    MRI_train_loader = DataLoader(full_dataset_train,
                                  batch_size= 1, 
                                  shuffle=False, 
                                  num_workers=0
                                  )

    All_skeletons = np.zeros((len(MRI_train_loader), ndiscs, 2))
    Joint_counter = np.zeros((ndiscs, 1))
    for i, (input, target, vis) in enumerate(MRI_train_loader):
        target = target.numpy()
        mask = np.zeros((target.shape[2], target.shape[3]))
        for idc in range(target.shape[1]):
            mask += target[0, idc]
        mask = np.uint8(np.where(mask>0, 1, 0))
        #mask = np.rot90(mask)
        num_labels, labels_im, states, centers = cv2.connectedComponentsWithStats(mask)
        centers = [t[::-1] for t in centers]
        skelet = np.zeros((ndiscs, 2))
        skelet[0:len(centers)-1] = centers[1:]
        Normjoint = np.linalg.norm(skelet[0]-skelet[4])
        for idx in range(1, len(centers)-1):
            skelet[idx] = (skelet[idx] - skelet[0]) / Normjoint

        skelet[0] *= 0
        
        All_skeletons[i] = skelet
        Joint_counter[0:len(centers)-1] += 1
        
    Skelet = np.sum(All_skeletons, axis= 0)   
    Joint_counter[Joint_counter==0]=1  # To avoid dividing by zero
    Skelet /= Joint_counter

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    np.save(os.path.join(out_dir, f'{args.contrasts}_Skelet_ndiscs_{ndiscs}.npy'), Skelet)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training hourglass network')
    
    ## Parameters
    parser.add_argument('--datapath', type=str, required=True,
                        help='Path to trainset')
    parser.add_argument('-c', '--contrast', type=str, metavar='N', required=True,
                        help='MRI contrast')
    parser.add_argument('--ndiscs', type=int, required=True,
                        help='Number of discs to detect')
    
    create_skeleton(parser.parse_args())  # Create skeleton file to improve hourglass accuracy during testing
