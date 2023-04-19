import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from spinalcordtoolbox.image import Image
import csv

from src.dlh.utils.metrics import compute_L2_error, compute_z_error, false_pos, false_neg
from src.dlh.utils.test_utils import CONTRAST, visualize_discs, str2array, check_missing_discs

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
    
    # Extract processed subjects --> subjects with a ground truth
    # line = subject_name contrast disc_num ground_truth_coord sct_label_vertebrae_coord hourglass_coord spinenet_coord
    processed_subjects = []
    for line in split_lines[1:]:
        if (line[0] not in processed_subjects) and (line[1]==contrast) and (line[3]!='None'):
            processed_subjects.append(line[0])
    
    # Initialize metrics
    methods_results = {}
    
    ## Init total to result dict
    methods_results['total'] = {}
    
    # Init L2 error
    methods_results['total']['l2_mean_hg'] = 0
    methods_results['total']['l2_mean_sct'] = 0
    methods_results['total']['l2_mean_spn'] = 0
    methods_results['total']['l2_std_hg'] = 0
    methods_results['total']['l2_std_sct'] = 0
    methods_results['total']['l2_std_spn'] = 0
    
    # Init Z error
    methods_results['total']['z_mean_hg'] = 0
    methods_results['total']['z_mean_sct'] = 0
    methods_results['total']['z_mean_spn'] = 0
    methods_results['total']['z_std_hg'] = 0
    methods_results['total']['z_std_sct'] = 0
    methods_results['total']['z_std_spn'] = 0
    
    # Init false positive rate
    methods_results['total']['FPR_hg'] = 0
    methods_results['total']['FPR_sct'] = 0
    methods_results['total']['FPR_spn'] = 0
    
    # Init false negative rate
    methods_results['total']['FNR_hg'] = 0
    methods_results['total']['FNR_sct'] = 0
    methods_results['total']['FNR_spn'] = 0
    
    for subject in processed_subjects:
        # Extract str coords and convert to numpy array, None stands for fail detections
        # line = subject_name contrast disc_num ground_truth_coord sct_label_vertebrae_coord hourglass_coord spinenet_coord
        discs_list = np.extract(split_lines[:,0] == subject,split_lines[:,2]).astype(int)
        sct_coords_list = str2array(np.extract(split_lines[:,0] == subject,split_lines[:,4]))
        hg_coords_list = str2array(np.extract(split_lines[:,0] == subject,split_lines[:,5]))
        gt_coords_list = str2array(np.extract(split_lines[:,0] == subject,split_lines[:,3])) 
        spn_coords_list = str2array(np.extract(split_lines[:,0] == subject,split_lines[:,6])) 
        
        # Check for missing ground truth (only ground truth detections are considered as real discs)
        _, gt_missing_discs = check_missing_discs(gt_coords_list) # Numpy array of coordinates without missing detections + list of missing discs
         
        # Check for missing discs predictions
        sct_discs_list, sct_missing_discs = check_missing_discs(sct_coords_list) # Numpy array of coordinates without missing detections + list of missing discs
        hg_discs_list, hg_missing_discs = check_missing_discs(hg_coords_list) # Numpy array of coordinates without missing detections + list of missing discs
        spn_discs_list, spn_missing_discs = check_missing_discs(spn_coords_list) # Numpy array of coordinates without missing detections + list of missing discs
        
        # Calculate total prediction and true number of discs
        total_discs = discs_list.shape[0] - gt_missing_discs.shape[0]
        total_pred_sct = discs_list.shape[0] - sct_missing_discs.shape[0]
        total_pred_hg = discs_list.shape[0] - hg_missing_discs.shape[0]
        total_pred_spn = discs_list.shape[0] - spn_missing_discs.shape[0]
        
        # Visualize discs on image
        if args.datapath != None:
            img_3D = Image(os.path.join(datapath, subject, f'{subject}_{contrast}.nii.gz')).data
            shape = img_3D.shape
            img_2D = np.rot90(img_3D[shape[0]//2, :, :]) # Rotate image to good position
            if sct_discs_list.size != 0: # Check if not empty
                visualize_discs(input_img=img_2D, coords_list=sct_discs_list, out_path=os.path.join(output_folder, f'{subject}_sct.png'))
            if hg_discs_list.size != 0: # Check if not empty
                visualize_discs(input_img=img_2D, coords_list=hg_discs_list, out_path=os.path.join(output_folder, f'{subject}_hg.png'))
            if spn_discs_list.size != 0: # Check if not empty
                visualize_discs(input_img=img_2D, coords_list=spn_discs_list, out_path=os.path.join(output_folder, f'{subject}_spn.png'))
            
        # Concatenate all the missing discs to compute metrics
        sct_and_gt_missing_discs = np.unique(np.append(sct_missing_discs, gt_missing_discs))
        hg_and_gt_missing_discs = np.unique(np.append(hg_missing_discs, gt_missing_discs))
        spn_and_gt_missing_discs = np.unique(np.append(spn_missing_discs, gt_missing_discs))
        sct_and_gt_missing_idx = np.in1d(discs_list, sct_and_gt_missing_discs) # get discs idx
        hg_and_gt_missing_idx = np.in1d(discs_list, hg_and_gt_missing_discs) # get discs idx
        spn_and_gt_missing_idx = np.in1d(discs_list, spn_and_gt_missing_discs) # get discs idx

        # Keep only coordinates that are present in both ground truth and prediction
        sct_coords_list = np.array(sct_coords_list[~sct_and_gt_missing_idx].tolist())
        hg_coords_list = np.array(hg_coords_list[~hg_and_gt_missing_idx].tolist())
        spn_coords_list = np.array(spn_coords_list[~spn_and_gt_missing_idx].tolist())
        
        # Add subject to result dict
        methods_results[subject] = {}
        
        #-------------------------#
        # Compute L2 error
        #-------------------------#
        l2_hourglass = compute_L2_error(gt=np.array(gt_coords_list[~hg_and_gt_missing_idx].tolist()), pred=hg_coords_list) # gt_coords_list[~hg_and_gt_missing_idx] to keep only coordinates that are present in both ground truth and prediction
        l2_sct = compute_L2_error(gt=np.array(gt_coords_list[~sct_and_gt_missing_idx].tolist()), pred=sct_coords_list) 
        l2_spn = compute_L2_error(gt=np.array(gt_coords_list[~spn_and_gt_missing_idx].tolist()), pred=spn_coords_list) 
        
        # Compute L2 error mean and std
        l2_hourglass_mean = np.mean(l2_hourglass) if l2_hourglass.size != 0 else 0
        l2_sct_mean = np.mean(l2_sct) if l2_sct.size != 0 else 0
        l2_spn_mean = np.mean(l2_spn) if l2_spn.size != 0 else 0
        l2_hourglass_std = np.std(l2_hourglass) if l2_hourglass.size != 0 else 0
        l2_sct_std = np.std(l2_sct) if l2_sct.size != 0 else 0
        l2_spn_std = np.std(l2_spn) if l2_spn.size != 0 else 0
        
        #--------------------------------#
        # Compute Z error
        #--------------------------------#
        z_err_hourglass = compute_z_error(gt=np.array(gt_coords_list[~hg_and_gt_missing_idx].tolist()), pred=hg_coords_list) # gt_coords_list[~hg_and_gt_missing_idx] to keep only coordinates that are present in both ground truth and prediction
        z_err_sct = compute_z_error(gt=np.array(gt_coords_list[~sct_and_gt_missing_idx].tolist()), pred=sct_coords_list)
        z_err_spn = compute_z_error(gt=np.array(gt_coords_list[~spn_and_gt_missing_idx].tolist()), pred=spn_coords_list)
        
        # Compute z error mean and std
        z_err_hourglass_mean = np.mean(z_err_hourglass) if z_err_hourglass.size != 0 else 0
        z_err_sct_mean = np.mean(z_err_sct) if z_err_sct.size != 0 else 0
        z_err_spn_mean = np.mean(z_err_spn) if z_err_spn.size != 0 else 0
        z_err_hourglass_std = np.std(z_err_hourglass) if z_err_hourglass.size != 0 else 0
        z_err_sct_std = np.std(z_err_sct) if z_err_sct.size != 0 else 0
        z_err_spn_std = np.std(z_err_spn) if z_err_spn.size != 0 else 0
        
        #-----------------------------------#
        # Compute false positive rate (FPR)
        #-----------------------------------#
        sct_pred_discs = discs_list[~np.in1d(discs_list, sct_missing_discs)]
        hg_pred_discs = discs_list[~np.in1d(discs_list, hg_missing_discs)]
        spn_pred_discs = discs_list[~np.in1d(discs_list, spn_missing_discs)]
    
        FP_sct, FP_list_sct = false_pos(missing_gt=gt_missing_discs, discs_pred=sct_pred_discs)
        FP_hg, FP_list_hg = false_pos(missing_gt=gt_missing_discs, discs_pred=hg_pred_discs)
        FP_spn, FP_list_spn = false_pos(missing_gt=gt_missing_discs, discs_pred=spn_pred_discs)
        
        FPR_sct = FP_sct/total_pred_sct if total_pred_sct != 0 else 0
        FPR_hg = FP_hg/total_pred_hg if total_pred_hg != 0 else 0
        FPR_spn = FP_spn/total_pred_spn if total_pred_spn != 0 else 0
        
        #-----------------------------------#
        # Compute false negative rate (FNR)
        #-----------------------------------#
        FN_sct, FN_list_sct = false_neg(missing_gt=gt_missing_discs, missing_pred=sct_missing_discs)
        FN_hg, FN_list_hg = false_neg(missing_gt=gt_missing_discs, missing_pred=hg_missing_discs)
        FN_spn, FN_list_spn = false_neg(missing_gt=gt_missing_discs, missing_pred=spn_missing_discs)
        
        FNR_sct = FN_sct/total_pred_sct if total_pred_sct != 0 else 1
        FNR_hg = FN_hg/total_pred_hg if total_pred_hg != 0 else 1
        FNR_spn = FN_spn/total_pred_spn if total_pred_spn != 0 else 1
        
        ###################################
        # Add computed metrics to subject #
        ###################################
        
        # Add L2 error
        methods_results[subject]['l2_mean_hg'] = l2_hourglass_mean
        methods_results[subject]['l2_mean_sct'] = l2_sct_mean
        methods_results[subject]['l2_mean_spn'] = l2_spn_mean
        methods_results[subject]['l2_std_hg'] = l2_hourglass_std
        methods_results[subject]['l2_std_sct'] = l2_sct_std
        methods_results[subject]['l2_std_spn'] = l2_spn_std
        
        # Add Z error
        methods_results[subject]['z_mean_hg'] = z_err_hourglass_mean
        methods_results[subject]['z_mean_sct'] = z_err_sct_mean
        methods_results[subject]['z_mean_spn'] = z_err_spn_mean
        methods_results[subject]['z_std_hg'] = z_err_hourglass_std
        methods_results[subject]['z_std_sct'] = z_err_sct_std
        methods_results[subject]['z_std_spn'] = z_err_spn_std
        
        # Add false positive rate
        methods_results[subject]['FP_list_hg'] = FP_list_hg
        methods_results[subject]['FP_list_sct'] = FP_list_sct
        methods_results[subject]['FP_list_spn'] = FP_list_spn
        methods_results[subject]['FPR_hg'] = FPR_hg
        methods_results[subject]['FPR_sct'] = FPR_sct
        methods_results[subject]['FPR_spn'] = FPR_spn
        
        # Add false negative rate
        methods_results[subject]['FN_list_hg'] = FN_list_hg
        methods_results[subject]['FN_list_sct'] = FN_list_sct
        methods_results[subject]['FN_list_spn'] = FN_list_spn
        methods_results[subject]['FNR_hg'] = FNR_hg
        methods_results[subject]['FNR_sct'] = FNR_sct
        methods_results[subject]['FNR_spn'] = FNR_spn
        
        methods_results[subject]['tot_discs'] = total_discs
        methods_results[subject]['tot_pred_sct'] = total_pred_sct
        methods_results[subject]['tot_pred_hg'] = total_pred_hg
        methods_results[subject]['tot_pred_spn'] = total_pred_spn
    
        #####################################
        # Add total mean of computed metrics#
        #####################################
        
        total = len(processed_subjects)
        
        # Add L2 error
        methods_results['total']['l2_mean_hg'] += l2_hourglass_mean/total
        methods_results['total']['l2_mean_sct'] += l2_sct_mean/total
        methods_results['total']['l2_mean_spn'] += l2_spn_mean/total
        methods_results['total']['l2_std_hg'] += l2_hourglass_std/total
        methods_results['total']['l2_std_sct'] += l2_sct_std/total
        methods_results['total']['l2_std_spn'] += l2_spn_std/total
        
        # Add Z error
        methods_results['total']['z_mean_hg'] += z_err_hourglass_mean/total
        methods_results['total']['z_mean_sct'] += z_err_sct_mean/total
        methods_results['total']['z_mean_spn'] += z_err_spn_mean/total
        methods_results['total']['z_std_hg'] += z_err_hourglass_std/total
        methods_results['total']['z_std_sct'] += z_err_sct_std/total
        methods_results['total']['z_std_spn'] += z_err_spn_std/total
        
        # Add false positive rate
        methods_results['total']['FPR_hg'] += FPR_hg/total
        methods_results['total']['FPR_sct'] += FPR_sct/total
        methods_results['total']['FPR_spn'] += FPR_spn/total
        
        # Add false negative rate
        methods_results['total']['FNR_hg'] = FNR_hg/total
        methods_results['total']['FNR_sct'] = FNR_sct/total
        methods_results['total']['FNR_spn'] = FNR_spn/total
    
    if args.create_csv:
        # Get fields for csv conversion    
        fields = ['subject'] + [key for key in methods_results[subject].keys()]
        
        csv_path = txt_file_path.replace('discs_coords.txt', 'computed_metrics.csv')
        with open(csv_path, "w") as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            for k,d in sorted(methods_results.items()):
                w.writerow(mergedict({'subject': k},d))
        
    save_graphs(output_folder, methods_results)
    return

def mergedict(a,b):
    a.update(b)
    return a

def save_graphs(output_folder, methods_results):
    del methods_results['total']
    subjects, subject_metrics = np.array(list(methods_results.keys())), list(methods_results.values())
    metrics_name = np.array(list(subject_metrics[0].keys()))
    metrics_values = np.array([list(sub_metrics.values()) for sub_metrics in subject_metrics])
    
        
    # Save L2 error
    l2_mean_hg_idx = np.where(metrics_name == 'l2_mean_hg')[0][0]
    l2_mean_sct_idx = np.where(metrics_name == 'l2_mean_sct')[0][0]
    l2_mean_spn_idx = np.where(metrics_name == 'l2_mean_spn')[0][0]
    out_path = os.path.join(output_folder,'l2_error.png')
    save_bar(subjects, 
             y1=metrics_values[:,l2_mean_hg_idx], 
             y2=metrics_values[:,l2_mean_sct_idx], 
             y3=metrics_values[:,l2_mean_spn_idx], 
             output_path=out_path, 
             x_axis='Subjects', 
             y_axis='L2 error (pixels)', 
             label1 ='hourglass_network', 
             label2 ='sct_label_vertebrae', 
             label3 ='spinenetv2_label_vertebrae'
             )
    
    # Save z error
    z_mean_hg_idx = np.where(metrics_name == 'z_mean_hg')[0][0]
    z_mean_sct_idx = np.where(metrics_name == 'z_mean_sct')[0][0]
    z_mean_spn_idx = np.where(metrics_name == 'z_mean_spn')[0][0]
    out_path = os.path.join(output_folder,'z_error.png')
    save_bar(subjects,
             y1=metrics_values[:,z_mean_hg_idx],
             y2=metrics_values[:,z_mean_sct_idx],
             y3=metrics_values[:,z_mean_spn_idx],
             output_path=out_path, x_axis='Subjects', 
             y_axis='z error (pixels)', 
             label1 ='hourglass_network', 
             label2 ='sct_label_vertebrae', 
             label3 ='spinenetv2_label_vertebrae'
             )

    
def save_bar(x, y1, y2, y3, output_path, x_axis='Subjects', y_axis='L2_error (pixels)', label1 ='Hourglass_network', label2 ='sct_label_vertebrae', label3 ='spinenetv2_label_vertebrae'):
    # set width of bar
    barWidth = 0.25
    plt.figure(figsize =(len(x), 10))
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.05, right=0.95)
        
    # Set position of bar on X axis
    br1 = np.arange(len(x))
    br2 = [x + barWidth for x in br1]
    br3 = [x - barWidth for x in br1]
    
    # Make the plot        
    plt.bar(br1, y1, color='b', width = barWidth, edgecolor ='grey', label=label1)
    plt.bar(br2, y2, color='r', width = barWidth, edgecolor ='grey', label=label2)
    plt.bar(br3, y3, color='g', width = barWidth, edgecolor ='grey', label=label3)
    plt.title(y_axis, fontweight ='bold', fontsize = 50)
    
    # Create axis and adding Xticks
    plt.xlabel(x_axis, fontweight ='bold', fontsize = 30)
    plt.ylabel(y_axis, fontweight ='bold', fontsize = 30)
    plt.xticks([r for r in range(len(x))], x, rotation=70)
    
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