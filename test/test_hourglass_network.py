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
from utils.test_utils import CONTRAST, extract_skeleton, best_disc_association, swap_y_origin, coord2list, project_on_spinal_cord 

#---------------------------Test Hourglass Network----------------------------
def test_hourglass(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contrast = CONTRAST[args.contrast]
    ndiscs = args.ndiscs
    origin_data = args.sct_datapath
    txt_file = args.out_txt_file
    
    print('load images')               
    with open(args.hg_datapath, 'rb') as file_pi:
        full = pickle.load(file_pi)            
    
    full[0] = full[0][:, :, :, :, 0]
    
    print('retrieving ground truth coordinates')
    norm_mean_skeleton = np.load(os.path.join(os.path.dirname(args.hg_datapath), f'{contrast}_Skelet_ndiscs_{ndiscs}.npy'))
    
    # Initialize metrics
    metrics = dict()
    metrics['distance_l2'] = []
    metrics['zdis'] = []
    metrics['faux_pos'] = []
    metrics['faux_neg'] = []
    metrics['tot'] = []
    
    # Load network weights
    if args.att:
        model = atthg(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=ndiscs)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(f'weights/model_{args.contrast}_att_stacks_{args.stacks}_ndiscs_{ndiscs}', map_location='cpu')['model_weights'])
    else:
        model = hg(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=ndiscs)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(f'weights/model_{args.contrast}_stacks_{args.stacks}_ndiscs_{ndiscs}', map_location='cpu')['model_weights'])

    # Create Dataloader
    full_dataset_test = image_Dataset(image_paths=full[0],
                                      target_paths=full[1],
                                      num_channel=ndiscs, 
                                      gt_coords=full[2], 
                                      subject_names=full[3], 
                                      use_flip = False
                                      ) 
    MRI_test_loader   = DataLoader(full_dataset_test, batch_size= 1, shuffle=False, num_workers=0)
    model.eval()
    
    # Load disc_coords txt file
    with open(txt_file,"r") as f:  # Checking already processed subjects from txt file
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
    
    # Extract discs coordinates from the test set
    for i, (input, target, vis, gt_coord, subject_name) in enumerate(MRI_test_loader): # subject_name
        input, target = input.to(device), target.to(device, non_blocking=True)
        output = model(input) 
        output = output[-1]
        original_img = np.rot90(full[0][i])  # rotate original input image to normal position 
        subject_name = subject_name[0]
        
        prediction = extract_skeleton(input, output, target, norm_mean_skeleton, ndiscs, Flag_save = True)
        prediction = np.sum(prediction[0], axis = 0)
        prediction = cv2.resize(prediction, (original_img.shape[0], original_img.shape[1]), interpolation=cv2.INTER_NEAREST)
        num_labels, labels_im, states, centers = cv2.connectedComponentsWithStats(np.uint8(np.where(prediction>0, 255, 0)))
        
        
        # Extract prediction and ground truth
        pred = centers[1:] #0 for background
        pred = swap_y_origin(coords=pred, img_shape=original_img.shape)  # Move y origin to the bottom of the image
        gt_coord = np.array(torch.tensor(gt_coord).tolist())
        
        # Project coordinate onto the spinal cord centerline 
        seg_path = os.path.join(origin_data, subject_name, f'{subject_name}_{contrast}_seg.nii.gz' )
        pred = project_on_spinal_cord(coords=pred, seg_path=seg_path, disc_num=False, proj_2d=True)
        gt_coord = project_on_spinal_cord(coords=gt_coord, seg_path=seg_path, disc_num=True, proj_2d=False)
        
        # Rearrange coordinates
        pred = coord2list(pred[~pred[:, 1].argsort()])  # Sorting predictions according to first coordinate and swap to coords convention [x, y] --> [lines(y), columns(x)]
        gt_coord = np.transpose(np.array([gt_coord[:,2],gt_coord[:,1],gt_coord[:,-1]])) # Using same format as prediction + discs label

        # Get best association between pred and gt
        # TODO - Extract discs numbers from prediction to avoid this step
        pred, gt = best_disc_association(pred=pred, gt=gt_coord)
        
        # Write coordinates in txt file
        # Edit txt_file --> line = subject_name contrast disc_num ground_truth_coord sct_label_vertebrae_coord hourglass_coord
        subject_index = np.where((np.array(split_lines)[:,0] == subject_name) & (np.array(split_lines)[:,1] == contrast))  
        start_index = subject_index[0][0]  # Getting the first line in the txt file
        last_index = subject_index[0][-1]  # Getting the last line for the subject in the txt file
        max_ref_disc = int(split_lines[last_index][2])  # Getting the last refferenced disc num
        for i in range(len(pred)):
            pred_coord = pred[i] if pred[i]!=0 else 'None'
            gt_coord = gt[i] if gt[i]!=0 else 'None'
            disc_num = i + 1
            if disc_num > max_ref_disc:
                print('More discs found')
                print('Disc number', disc_num)
                new_line = [subject_name, contrast, str(disc_num), 'None', 'None', 'None\n']
                if pred_coord != 'None':
                    new_line[5] = '[' + str(pred_coord[0]) + ',' + str(pred_coord[1]) + ']\n'
                elif gt_coord == 'None':
                    new_line[5] = 'None\n'
                else:
                    new_line[5] = 'Fail\n'
                if gt_coord != 'None':
                    new_line[3] = '[' + str(gt_coord[0]) + ',' + str(gt_coord[1]) + ']'
                else:
                    new_line[3] = 'None'
                last_index += 1
                split_lines.insert(last_index, new_line) # Add new disc detection to txt_file lines
                max_ref_disc = disc_num
            else:
                if pred_coord != 'None':
                    split_lines[start_index + i][5] = '[' + str(pred_coord[0]) + ',' + str(pred_coord[1]) + ']\n'
                elif gt_coord == 'None':
                    split_lines[start_index + i][5] = 'None\n'
                else:
                    split_lines[start_index + i][5] = 'Fail\n'
                if gt_coord != 'None':
                    split_lines[start_index + i][3] = '[' + str(gt_coord[0]) + ',' + str(gt_coord[1]) + ']'
                else:
                    split_lines[start_index + i][3] = 'None'
                
    for num in range(len(split_lines)):
        split_lines[num] = ' '.join(split_lines[num])
        
    with open(txt_file,"w") as f:
        f.writelines(split_lines)
        