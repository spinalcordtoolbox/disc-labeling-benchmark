import os
import cv2
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader 

from dlh.models.hourglass import hg
from dlh.models.atthourglass import atthg
from dlh.utils.train_utils import image_Dataset
from dlh.utils.test_utils import CONTRAST, extract_skeleton, swap_y_origin, coord2list, project_on_spinal_cord, edit_subject_lines_txt_file

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
        model.load_state_dict(torch.load(f'src/dlh/weights/model_{args.contrast}_att_stacks_{args.stacks}_ndiscs_{ndiscs}', map_location='cpu')['model_weights'])
    else:
        model = hg(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=ndiscs)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(f'src/dlh/weights/model_{args.contrast}_stacks_{args.stacks}_ndiscs_{ndiscs}', map_location='cpu')['model_weights'])

    # Create Dataloader
    full_dataset_test = image_Dataset(image_paths=full[0],
                                      target_paths=full[1],
                                      num_channel=ndiscs, 
                                      gt_coords=full[2], 
                                      subject_names=full[3], 
                                      use_flip = False
                                      ) 
    MRI_test_loader   = DataLoader(full_dataset_test, 
                                   batch_size= 1, 
                                   shuffle=False, 
                                   num_workers=0
                                   )
    model.eval()
    
    # Load disc_coords txt file
    with open(txt_file,"r") as f:  # Checking already processed subjects from txt file
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
    
    # Extract discs coordinates from the test set
    print('Processing with hourglass')
    for i, (input, target, vis, gt_coord, subject_name) in enumerate(MRI_test_loader): # subject_name
        input, target = input.to(device), target.to(device, non_blocking=True)
        output = model(input) 
        output = output[-1]
        original_img = np.rot90(full[0][i])  # rotate original input image to normal position 
        subject_name = subject_name[0]
        
        prediction, pred_discs_coords = extract_skeleton(input, output, target, norm_mean_skeleton, ndiscs, Flag_save = True)
        
        # Convert pred_discs_coords to original image size
        pred_shape = prediction[0,0].shape 
        original_shape = original_img.shape 
        pred = np.array([[(round(coord[1])/pred_shape[1])*original_shape[1], (round(coord[0])/pred_shape[0])*original_shape[0], int(disc_num)] for disc_num, coord in pred_discs_coords[0].items()]).astype(int)
       
        # Extract prediction and ground truth
        pred = swap_y_origin(coords=pred, img_shape=original_img.shape)  # Move y origin to the bottom of the image like Niftii convention
        gt_coord = np.array(torch.tensor(gt_coord).tolist())
        
        # Project coordinate onto the spinal cord centerline 
        seg_path = os.path.join(origin_data, subject_name, f'{subject_name}_{contrast}_seg.nii.gz' )
        pred = project_on_spinal_cord(coords=pred, seg_path=seg_path, disc_num=True, proj_2d=True)
        gt_coord = project_on_spinal_cord(coords=gt_coord, seg_path=seg_path, disc_num=True, proj_2d=False)
        
        # Rearrange coordinates
        pred[:, :2] = coord2list(pred[~pred[:, 1].argsort(), :2])  # Sorting predictions according to first coordinate and swap to coords convention [x, y] --> [lines(y), columns(x)]
        pred = pred.astype(int)
        gt_coord = np.transpose(np.array([gt_coord[:,2].astype(int),gt_coord[:,1].astype(int),gt_coord[:,-1].astype(int)])) # Using same format as prediction + discs label and convert to integer
        
        # Edit coordinates in txt file
        # line = subject_name contrast disc_num gt_coords sct_label_vertebrae_coords hourglass_coords spinenet_coords
        split_lines = edit_subject_lines_txt_file(coords=gt_coord, txt_lines=split_lines, subject_name=subject_name, contrast=contrast, method_name='gt_coords')
        split_lines = edit_subject_lines_txt_file(coords=pred, txt_lines=split_lines, subject_name=subject_name, contrast=contrast, method_name='hourglass_coords')
                
    for num in range(len(split_lines)):
        split_lines[num] = ' '.join(split_lines[num])
        
    with open(txt_file,"w") as f:
        f.writelines(split_lines)

