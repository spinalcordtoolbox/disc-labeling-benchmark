import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from spinalcordtoolbox.image import Image
import csv
import pandas as pd
import dataframe_image as dfi

parent_dir = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(parent_dir)

from utils.metrics import compute_L2_error, compute_z_error, false_pos, false_neg
from utils.test_utils import CONTRAST, visualize_discs, str2array, check_missing_discs

def compare_methods(args):
    if args.datapath != None:
        datapath = args.datapath
    contrast = CONTRAST[args.modality]
    txt_file_path = args.input_txt_file
    output_folder = os.path.join(args.output_folder, f'out_{contrast}')
    
    # Create output folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    # Load disc_coords txt file
    with open(txt_file_path,"r") as f:  # Checking already processed subjects from txt file
        file_lines = f.readlines()
        split_lines = np.array([line.split(' ') for line in file_lines])
    
    # Extract subjects processed by sct and hourglass method
    processed_subjects = []
    for line in split_lines[1:]:
        if (line[0] not in processed_subjects) and (line[1]==contrast) and (line[5]!='None\n'):
            processed_subjects.append(line[0])
    
    # Initialize metrics
    methods_results = {} 
    l2_mean_hg = []
    l2_std_hg = []
    l2_mean_sct = []
    l2_std_sct = []
    z_mean_hg = []
    z_std_hg = []
    z_mean_sct = []
    z_std_sct = []
    FP_list_hg_total = []
    FP_list_sct_total = []
    FPR_hg_total = []
    FPR_sct_total = []
    FN_list_hg_total = []
    FN_list_sct_total = []
    FNR_hg_total = []
    FNR_sct_total = []
    tot_discs = []
    tot_pred_hg = []
    tot_pred_sct =[]
    for subject in processed_subjects:
        # Extract str coords and convert to numpy array, None stands for fail detections
        discs_list = np.extract(split_lines[:,0] == subject,split_lines[:,2]).astype(int)
        sct_coords_list = str2array(np.extract(split_lines[:,0] == subject,split_lines[:,3]))
        hg_coords_list = str2array(np.extract(split_lines[:,0] == subject,split_lines[:,4]))
        gt_coords_list = str2array(np.extract(split_lines[:,0] == subject,split_lines[:,5])) 
        
        # Check for missing ground truth (only ground truth detections are considered as real discs)
        _, gt_missing_discs = check_missing_discs(gt_coords_list) # Numpy array of coordinates without missing detections + list of missing discs
         
        # Check for missing discs predictions
        sct_discs_list, sct_missing_discs = check_missing_discs(sct_coords_list) # Numpy array of coordinates without missing detections + list of missing discs
        hg_discs_list, hg_missing_discs = check_missing_discs(hg_coords_list) # Numpy array of coordinates without missing detections + list of missing discs
        
        # Calculate total prediction and true number of discs
        total_discs = discs_list.shape[0] - gt_missing_discs.shape[0]
        total_pred_sct = discs_list.shape[0] - sct_missing_discs.shape[0]
        total_pred_hg = discs_list.shape[0] - hg_missing_discs.shape[0]
        
        # Visualize discs on image
        if args.datapath != None:
            img_3D = Image(os.path.join(datapath, subject, f'{subject}_{contrast}.nii.gz')).data
            shape = img_3D.shape
            img_2D = np.rot90(img_3D[shape[0]//2, :, :]) # Rotate image to good position
            if sct_discs_list.size != 0: # Check if not empty
                visualize_discs(input_img=img_2D, coords_list=sct_discs_list, out_path=os.path.join(output_folder, f'{subject}_sct.png'))
            if hg_discs_list.size != 0: # Check if not empty
                visualize_discs(input_img=img_2D, coords_list=hg_discs_list, out_path=os.path.join(output_folder, f'{subject}_hg.png'))
            
        # Concatenate all the missing discs to compute metrics
        sct_and_gt_missing_discs = np.unique(np.append(sct_missing_discs, gt_missing_discs))
        hg_and_gt_missing_discs = np.unique(np.append(hg_missing_discs, gt_missing_discs))
        sct_and_gt_missing_idx = np.in1d(discs_list, sct_and_gt_missing_discs) # get discs idx
        hg_and_gt_missing_idx = np.in1d(discs_list, hg_and_gt_missing_discs) # get discs idx

        # Keep only coordinates that are present in both ground truth and prediction
        sct_coords_list = np.array(sct_coords_list[~sct_and_gt_missing_idx].tolist())
        hg_coords_list = np.array(hg_coords_list[~hg_and_gt_missing_idx].tolist())
        
        # Add subject to result dict
        methods_results[subject] = {}
        
        #-------------------------#
        # Compute L2 error
        #-------------------------#
        L2_hourglass = compute_L2_error(gt=np.array(gt_coords_list[~hg_and_gt_missing_idx].tolist()), pred=hg_coords_list) # gt_coords_list[~hg_and_gt_missing_idx] to keep only coordinates that are present in both ground truth and prediction
        L2_sct = compute_L2_error(gt=np.array(gt_coords_list[~sct_and_gt_missing_idx].tolist()), pred=sct_coords_list) 
        
        # Compute L2 error mean and std
        L2_hourglass_mean = np.mean(L2_hourglass) if L2_hourglass.size != 0 else 0
        L2_sct_mean = np.mean(L2_sct) if L2_sct.size != 0 else 0
        L2_hourglass_std = np.std(L2_hourglass) if L2_hourglass.size != 0 else 0
        L2_sct_std = np.std(L2_sct) if L2_sct.size != 0 else 0
        
        #--------------------------------#
        # Compute Z error
        #--------------------------------#
        z_err_hourglass = compute_z_error(gt=np.array(gt_coords_list[~hg_and_gt_missing_idx].tolist()), pred=hg_coords_list) # gt_coords_list[~hg_and_gt_missing_idx] to keep only coordinates that are present in both ground truth and prediction
        z_err_sct = compute_z_error(gt=np.array(gt_coords_list[~sct_and_gt_missing_idx].tolist()), pred=sct_coords_list)
        
        # Compute z error mean and std
        z_err_hourglass_mean = np.mean(z_err_hourglass) if z_err_hourglass.size != 0 else 0
        z_err_sct_mean = np.mean(z_err_sct) if z_err_sct.size != 0 else 0
        z_err_hourglass_std = np.std(z_err_hourglass) if z_err_hourglass.size != 0 else 0
        z_err_sct_std = np.std(z_err_sct) if z_err_sct.size != 0 else 0
        
        #-----------------------------------#
        # Compute false positive rate (FPR)
        #-----------------------------------#
        sct_pred_discs = discs_list[~np.in1d(discs_list, sct_missing_discs)]
        hg_pred_discs = discs_list[~np.in1d(discs_list, hg_missing_discs)]
    
        FP_sct, FP_list_sct = false_pos(missing_gt=gt_missing_discs, discs_pred=sct_pred_discs)
        FP_hg, FP_list_hg = false_pos(missing_gt=gt_missing_discs, discs_pred=hg_pred_discs)
        
        FPR_sct = FP_sct/total_pred_sct if total_pred_sct != 0 else 0
        FPR_hg = FP_hg/total_pred_hg if total_pred_hg != 0 else 0
        
        #-----------------------------------#
        # Compute false negative rate (FNR)
        #-----------------------------------#
        FN_sct, FN_list_sct = false_neg(missing_gt=gt_missing_discs, missing_pred=sct_missing_discs)
        FN_hg, FN_list_hg = false_neg(missing_gt=gt_missing_discs, missing_pred=hg_missing_discs)
        
        FNR_sct = FN_sct/total_pred_sct if total_pred_sct != 0 else 1
        FNR_hg = FN_hg/total_pred_hg if total_pred_hg != 0 else 1
        
        ###################################
        # Add computed metrics to subject #
        ###################################
        
        # Add L2 error
        methods_results[subject]['l2_mean_hg'] = L2_hourglass_mean
        methods_results[subject]['l2_mean_sct'] = L2_sct_mean
        methods_results[subject]['l2_std_hg'] = L2_hourglass_std
        methods_results[subject]['l2_std_sct'] = L2_sct_std
        l2_mean_hg.append(L2_hourglass_mean)
        l2_mean_sct.append(L2_sct_mean)
        l2_std_hg.append(L2_hourglass_std)
        l2_std_sct.append(L2_sct_std)
        
        # Add Z error
        methods_results[subject]['z_mean_hg'] = z_err_hourglass_mean
        methods_results[subject]['z_mean_sct'] = z_err_sct_mean
        methods_results[subject]['z_std_hg'] = z_err_hourglass_std
        methods_results[subject]['z_std_sct'] = z_err_sct_std
        z_mean_hg.append(z_err_hourglass_mean)
        z_mean_sct.append(z_err_sct_mean)
        z_std_hg.append(z_err_hourglass_std)
        z_std_sct.append(z_err_sct_std)
        
        # Add false positive rate
        methods_results[subject]['FP_list_hg'] = FP_list_hg
        methods_results[subject]['FP_list_sct'] = FP_list_sct
        methods_results[subject]['FPR_hg'] = FPR_hg
        methods_results[subject]['FPR_sct'] = FPR_sct
        FP_list_hg_total.append(FP_list_hg)
        FP_list_sct_total.append(FP_list_sct)
        FPR_hg_total.append(FPR_hg)
        FPR_sct_total.append(FPR_sct)
        
        # Add false negative rate
        methods_results[subject]['FN_list_hg'] = FN_list_hg
        methods_results[subject]['FN_list_sct'] = FN_list_sct
        methods_results[subject]['FNR_hg'] = FNR_hg
        methods_results[subject]['FNR_sct'] = FNR_sct
        FN_list_hg_total.append(FN_list_hg)
        FN_list_sct_total.append(FN_list_sct)
        FNR_hg_total.append(FNR_hg)
        FNR_sct_total.append(FNR_sct)
        
        methods_results[subject]['tot_discs'] = total_discs
        methods_results[subject]['tot_pred_sct'] = total_pred_sct
        methods_results[subject]['tot_pred_hg'] = total_pred_hg
        tot_discs.append(total_discs)
        tot_pred_sct.append(total_pred_sct)
        tot_pred_hg.append(total_pred_hg)
    
    #####################################
    # Add total mean of computed metrics#
    #####################################
    
    # Add subject to result dict
    methods_results['total'] = {}
        
    # Add L2 error
    methods_results['total']['l2_mean_hg'] = sum(l2_mean_hg)/len(l2_mean_hg)
    methods_results['total']['l2_mean_sct'] = sum(l2_mean_sct)/len(l2_mean_sct)
    methods_results['total']['l2_std_hg'] = sum(l2_std_hg)/len(l2_std_hg)
    methods_results['total']['l2_std_sct'] = sum(l2_std_sct)/len(l2_std_sct)
    
    # Add Z error
    methods_results['total']['z_mean_hg'] = sum(z_mean_hg)/len(z_mean_hg)
    methods_results['total']['z_mean_sct'] = sum(z_mean_sct)/len(z_mean_sct)
    methods_results['total']['z_std_hg'] = sum(z_std_hg)/len(z_std_hg)
    methods_results['total']['z_std_sct'] = sum(z_std_sct)/len(z_std_sct)
    
    # Add false positive rate
    methods_results['total']['FPR_hg'] = sum(FPR_hg_total)/len(FPR_hg_total)
    methods_results['total']['FPR_sct'] = sum(FPR_sct_total)/len(FPR_sct_total)
    
    # Add false negative rate
    methods_results['total']['FNR_hg'] = sum(FNR_hg_total)/len(FNR_hg_total)
    methods_results['total']['FNR_sct'] = sum(FNR_sct_total)/len(FNR_sct_total)
    
    if args.create_csv:
        # Get fields for csv conversion    
        fields = ['subject'] + [key for key in methods_results[subject].keys()]
        
        csv_path = txt_file_path.replace('discs_coords.txt', 'computed_metrics.csv')
        with open(csv_path, "w") as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            for k,d in sorted(methods_results.items()):
                w.writerow(mergedict({'subject': k},d))
    
    #dfi.export(pd.DataFrame(data=dict['total'], index=[0]), os.path.join(output_folder,'total_metrics.png'))
    
    save_graphs(output_folder, processed_subjects, l2_mean_hg, l2_std_hg, l2_mean_sct, l2_std_sct, z_mean_hg, z_std_hg, z_mean_sct, z_std_sct, FP_list_hg_total, FP_list_sct_total, FPR_hg_total, FPR_sct_total, FN_list_hg_total, FN_list_sct_total, FNR_hg_total, FNR_sct_total, tot_discs, tot_pred_hg, tot_pred_sct)
    return

def mergedict(a,b):
    a.update(b)
    return a

def save_graphs(output_folder, subjects, l2_mean_hg, l2_std_hg, l2_mean_sct, l2_std_sct, z_mean_hg, z_std_hg, z_mean_sct, z_std_sct, FP_list_hg_total, FP_list_sct_total, FPR_hg_total, FPR_sct_total, FN_list_hg_total, FN_list_sct_total, FNR_hg_total, FNR_sct_total, tot_discs, tot_pred_hg, tot_pred_sct):
    # Save L2 error
    out_path = os.path.join(output_folder,'l2_error.png')
    save_bar(subjects, l2_mean_hg, l2_mean_sct, output_path=out_path, x_axis='Subjects', y_axis='L2 error (pixels)', label1 ='Hourglass_network', label2 ='sct_label_vertebrae')
    
    # Save z error
    out_path = os.path.join(output_folder,'z_error.png')
    save_bar(subjects, z_mean_hg, z_mean_sct, output_path=out_path, x_axis='Subjects', y_axis='z error (pixels)', label1 ='Hourglass_network', label2 ='sct_label_vertebrae')

    
def save_bar(x, y1, y2, output_path, x_axis='Subjects', y_axis='L2_error (pixels)', label1 ='Hourglass_network', label2 ='sct_label_vertebrae'):
    # set width of bar
    barWidth = 0.25
    plt.figure(figsize =(len(x), 10))
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.05, right=0.95)
        
    # Set position of bar on X axis
    br1 = np.arange(len(x))
    br2 = [x + barWidth for x in br1]
    
    # Make the plot        
    plt.bar(br1, y1, color='b', width = barWidth, edgecolor ='grey', label=label1)
    plt.bar(br2, y2, color='r', width = barWidth, edgecolor ='grey', label=label2)
    plt.title(y_axis, fontweight ='bold', fontsize = 50)
    
    # Create axis and adding Xticks
    plt.xlabel(x_axis, fontweight ='bold', fontsize = 30)
    plt.ylabel(y_axis, fontweight ='bold', fontsize = 30)
    plt.xticks([r + barWidth/2 for r in range(len(x))], x, rotation=70)
    
    # Save plot
    plt.legend()
    plt.savefig(output_path)

 
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute metrics on sct and hourglass disc estimation')
    
    parser.add_argument('--datapath', type=str, metavar='N', default=None,
                        help='Path to dataset cf gather_data format')
    parser.add_argument('-txt', '--input-txt-file', type=str, metavar='N', required=True,
                        help='Input txt file with the methods coordinates')
    parser.add_argument('-o', '--output-folder', type=str, metavar='N', required=True,
                        help='Output folder for created graphs') 
    parser.add_argument('-c', '--modality', type=str, metavar='N', required=True,
                        help='Data modality')
    parser.add_argument('-csv', '--create-csv', metavar='N', default=True,
                        help='Output csv file with computed metrics') 
    
    
    compare_methods(parser.parse_args())