import os
import sys
import cv2
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader 

parent_dir = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(parent_dir)

from models.hourglass import hg
from models.atthourglass import atthg
from utils.train_utils import image_Dataset
from utils.test_utils import CONTRAST, extract_skeleton, best_disc_association 

#---------------------------Test Hourglass Network----------------------------
def test_hourglass(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contrast = CONTRAST[args.contrast]
    txt_file = args.out_txt_file
    
    print('load images')               
    with open(args.hg_datapath, 'rb') as file_pi:
        full = pickle.load(file_pi)            
    
    full[0] = full[0][:, :, :, :, 0]
    
    print('retrieving ground truth coordinates')
    norm_mean_skeleton = np.load(os.path.join(os.path.dirname(args.hg_datapath), f'{contrast}_Skelet.npy'))
    
    # Initialize metrics
    metrics = dict()
    metrics['distance_l2'] = []
    metrics['zdis'] = []
    metrics['faux_pos'] = []
    metrics['faux_neg'] = []
    metrics['tot'] = []
    
    # Load network weights
    if args.att:
        model = atthg(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.ndiscs)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(f'weights/model_{args.contrast}_att_stacks_{args.stacks}_ndiscs_{args.ndiscs}', map_location='cpu')['model_weights'])
    else:
        model = hg(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.ndiscs)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(f'weights/model_{args.contrast}_stacks_{args.stacks}_ndiscs_{args.ndiscs}', map_location='cpu')['model_weights'])

    # Create Dataloader
    full_dataset_test = image_Dataset(image_paths=full[0],
                                      target_paths=full[1],
                                      num_channel=args.ndiscs, 
                                      gt_coords=full[2], 
                                      subject_names=full[3], 
                                      use_flip = False
                                      ) 
    MRI_test_loader   = DataLoader(full_dataset_test, batch_size= 1, shuffle=False, num_workers=0)
    model.eval()
    
    # Load disc_coords txt file
    with open(txt_file,"r") as f:  # Checking already processed subjects from coords.txt
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
    
    # Get the visualization results of the test set
    for i, (input, target, vis, gt_coord, subject_name) in enumerate(MRI_test_loader): # subject_name
        input, target = input.to(device), target.to(device, non_blocking=True)
        output = model(input) 
        output = output[-1]
        x      = full[0][i]
        
        print(subject_name)
        prediction = extract_skeleton(input, output, target, norm_mean_skeleton, Flag_save = True)
        prediction = np.sum(prediction[0], axis = 0)
        prediction = np.rot90(prediction,3)
        prediction = cv2.resize(prediction, (x.shape[0], x.shape[1]), interpolation=cv2.INTER_NEAREST)
        num_labels, labels_im, states, centers = cv2.connectedComponentsWithStats(np.uint8(np.where(prediction>0, 255, 0)))
        
        
        # Write the predicted and ground truth coordinates inside the discs_coords txt file
        pred = centers[1:] #0 for background
        pred = np.flip(pred[pred[:, 0].argsort()], axis=0)  # Sorting predictions according to first coordinate
        gt_coord = np.array(torch.tensor(gt_coord).tolist())
        gt_coord = np.transpose(np.array([gt_coord[:,2],gt_coord[:,1],gt_coord[:,-1]])) # Using same format as prediction + discs label
        subject_index = np.where((np.array(split_lines)[:,0] == subject_name[0]) & (np.array(split_lines)[:,1] == contrast))  
        start_index = subject_index[0][0]  # Getting the first line in the txt file
        
        pred, gt = best_disc_association(pred=pred, gt=gt_coord)
        for i in range(len(pred)):
            pred_coord = pred[i] if pred[i]!=0 else 'Fail'
            gt_coord = gt[i] if gt[i]!=0 else 'None'
            if pred_coord != 'Fail':
                split_lines[start_index + i][4] = '[' + str("{:.1f}".format(pred_coord[0])) + ',' + str("{:.1f}".format(pred_coord[1])) + ']'
            elif gt_coord == 'None':
                split_lines[start_index + i][4] = 'None'
            else:
                split_lines[start_index + i][4] = 'Fail'
            if gt_coord != 'None':
                split_lines[start_index + i][5] = '[' + str(gt_coord[0]) + ',' + str(gt_coord[1]) + ']' + '\n'
            else:
                split_lines[start_index + i][5] = 'None' + '\n'
                
        for num in range(len(split_lines)):
            file_lines[num] = ' '.join(split_lines[num])
            
        with open(txt_file,"w") as f:
            f.writelines(file_lines)  
        