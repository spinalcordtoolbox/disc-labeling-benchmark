import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd
import json

from bcm.utils.utils import SCT_CONTRAST, edit_metric_csv, fetch_img_and_seg_paths, fetch_subject_and_session #,visualize_discs
from bcm.utils.image import Image


def compare_methods(args):
    if args.config_data:
        config_data = json.load(open(args.config_data, "r"))

        img_paths, seg_paths = fetch_img_and_seg_paths(path_list=config_data['TESTING'], 
                                                    path_type=config_data['TYPE'],
                                                    seg_suffix='_seg-manual',
                                                    derivatives_path='derivatives/labels'
                                                    )
    txt_file_path = args.input_txt_file
    dataset = os.path.basename(txt_file_path).split('_')[0]
    output_folder = os.path.join(args.output_folder, f'out_{dataset}')
    computed_methods = args.computed_methods
    


    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load disc_coords txt file
    with open(txt_file_path,"r") as f:  # Checking already processed subjects from txt file
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
    
    ################################################################################################################################################################################################
    #Permet d'afficher l'intérieur du fichier pour vérifier que toutes les colonnes sont affichées
    #if split_lines:
    #    print("Première ligne du fichier :", split_lines[0])
    ################################################################################################################################################################################################


    # Extract processed subjects --> subjects with a ground truth
    # line = subject_name contrast num_disc gt_coords sct_discs_coords spinenet_coords hourglass_t1_coords hourglass_t2_coords hourglass_t1_t2_coords
    processed_subjects_dict = dict()
    for line in split_lines[1:]:
        gt_condition = (line[split_lines[0].index('gt_coords')]!='None') or not args.gt_exists # Process subject only if ground truth are available or not args.gt_exists
        if (line[split_lines[0].index('subject_name')] not in processed_subjects_dict.keys()) and gt_condition:
            processed_subjects_dict[line[split_lines[0].index('subject_name')]] = [line[split_lines[0].index('contrast')]]
        elif (line[split_lines[0].index('subject_name')] in processed_subjects_dict.keys()) and gt_condition:
            if (line[split_lines[0].index('contrast')] not in processed_subjects_dict[line[split_lines[0].index('subject_name')]]):
                processed_subjects_dict[line[split_lines[0].index('subject_name')]].append(line[split_lines[0].index('contrast')])
    #processed_subjects_dict = {subject1 : [T1w, T2w],
    #                           subject2 : [T2w],
    #                           subject3 : [T1w, T2w]}


    # Initialize metrics for each contrast
    methods_results = {}
    for contrasts in list(processed_subjects_dict.values()):
        for c in contrasts:
            if c not in methods_results.keys():
                methods_results[c] = {}
    
    nb_subjects = len(processed_subjects_dict.keys())
    for method in computed_methods:
        print(f"Computing method {method}")
        for subject, contrasts in processed_subjects_dict.items():
            for contrast in contrasts:
                methods_results[contrast], pred_discs_list = edit_metric_csv(methods_results[contrast], txt_lines=split_lines, subject_name=subject, contrast=contrast, method_name=method, nb_subjects=nb_subjects)

#             def create_binary_mask(coord_list, image_shape):
#                 # Initialize an empty binary mask
#                 binary_mask = np.zeros(image_shape, dtype=bool)
    
#                  # Set the pixels within the coordinate list to True
#                 for coord in coord_list:
#                     binary_mask[coord[0], coord[1]] = True
#             return binary_mask
#             ground_truth_mask = create_binary_mask(ground_truth_coords, image_shape)
#             predicted_mask = create_binary_mask(pred_discs_list, image_shape)

            # Visualize discs on image
            """
            if args.config_data:
                sub_name = f'{subject}_{contrast}'
                for img_path in img_paths:
                    if sub_name in img_path:
                        img_3D = Image(img_path).change_orientation('RSP').data
                        shape = img_3D.shape
                        img_2D = img_3D[shape[0]//2, :, :]
                        #if pred_discs_list.size != 0: # Check if not empty
                            #visualize_discs(input_img=img_2D, coords_list=pred_discs_list, out_path=os.path.join(output_folder, f'{sub_name}_{method}.png'))
            """
    
    if args.create_csv:
        for contrast in methods_results.keys():
            # Get fields for csv conversion
            sub_list = [sub for sub in methods_results[contrast].keys() if sub.startswith('sub')]
            fields = ['subject'] + [key for key in methods_results[contrast][sub_list[0]].keys()]
            
            csv_path = txt_file_path.replace('discs_coords.txt', f'computed_metrics_{contrast}.csv')
            with open(csv_path, "w") as f:
                w = csv.DictWriter(f, fields)
                w.writeheader()
                for k,d in sorted(methods_results[contrast].items()):
                    w.writerow(mergedict({'subject': k},d))
    
    # Remove unrelevant hourglass contrasts and shorten methods names
    for contrast in methods_results.keys():
        methods_plot = []
        for method in computed_methods:
            if 'hourglass' in method:
                if SCT_CONTRAST[contrast] in method:
                    methods_plot.append(method.split('_coords')[0]) # Remove '_coords' suffix
            elif 'nnunet' in method:
                methods_plot.append(f'{method}') 
            else:
                methods_plot.append(method.split('_coords')[0]) # Remove '_coords' suffix
        save_graphs(output_folder, methods_results[contrast], methods_plot, contrast)


        ## prevenir cas particulier nnunet garder numero apres coords

def mergedict(a,b):
    a.update(b)
    return a


def save_graphs(output_folder, methods_results, methods_list, contrast):
    # Isolate total dict 
    dict_total = methods_results['total']
    del methods_results['total']
            
    # Extract subjects and metrics
    subjects, subject_metrics = np.array(list(methods_results.keys())), list(methods_results.values())
    metrics_name = np.array(list(subject_metrics[0].keys())) 

    # Find the position of "tot" in metrics_name
    position_tot = [i for i, name in enumerate(metrics_name) if "tot" in name]
    # Remove values from metrics_name at the found positions
    metrics_name = [name for i, name in enumerate(metrics_name) if i not in position_tot]
    # metrics_name contains only the remaining names
    metrics_values = np.array([list(sub_metrics.values()) for sub_metrics in subject_metrics])
    # 2D Numpy array, where each row corresponds to a subject, and each column corresponds to a metric.

    # Delete columns corresponding to a name with "tot" in each row of the array
    metrics_values = np.delete(metrics_values, position_tot,axis=1)

    metrics_mean_list=[]
    metrics_std_list=[]
    metrics_list_bar= [] 
    metric_name_only_list = []
    number=[101,102,200,201] #nombre d'identification de l'entrainement pour les méthodes nnunet

    for method_name in methods_list:
        # Initialize lists for FPR and TPR values
        TPR_values = []
        FPR_values = []
        for sub_metrics in subject_metrics:
            for metric_name, metric_value in sub_metrics.items():
                if f"TPR_{method_name}" in metric_name:
                    TPR_values.append(metric_value)
                elif f"FPR_{method_name}" in metric_name:
                    FPR_values.append(metric_value)
        out_path = os.path.join(output_folder, f'ROC_{contrast}')
        save_ROC_curves(methods=methods_list, TPR_list= TPR_values, FPR_list= FPR_values, output_path=out_path,x_axis='True Positive Rate (TPR)', y_axis='False Positive Rate (FPR)')

    
    for metric_name in metrics_name:
         # Find the last "_mean" in the metric name
        last_mean_index = metric_name.rfind("_mean")
        # Find the last "_std" in the metric name
        last_std_index = metric_name.rfind("_std")
        metric_name_parts = metric_name.split("_", 1)
        metric_std = ''
        metric_mean=''

        if last_mean_index != -1:
            metric_name_only = metric_name[:last_mean_index] + "_mean"
            method_name = metric_name[last_mean_index + 6:]
            metric_mean = metric_name_only 
        elif last_std_index != -1:
            metric_name_only = metric_name[:last_std_index] + "_std"
            method_name = metric_name[last_std_index + 5:]
            metric_std = metric_name_only
        elif len(metric_name_parts) == 2:
            metric_name_only, method_name = metric_name_parts
        
        if not metric_name_only in metric_name_only_list:
            metric_name_only_list.append(metric_name_only)

        if metric_mean :
            metric_mean_name, mean = metric_mean.split("_", 1)
            if not metric_mean_name in metrics_mean_list:
                metrics_mean_list.append(metric_mean_name)
        
        if metric_std :
            metric_std_name, std = metric_std.split("_", 1)
            if not metric_std_name in metrics_std_list:
                metrics_std_list.append(metric_std_name)
        
        metrics_list_bar = metrics_mean_list 

        print("Metric name:", metric_name_only)
        print("Method name:", method_name)

   
    for name in metrics_list_bar:
        metric_values_list_bar = []
        for method in methods_list:
            method_mean_values = [] # cette ligne stock les moyennes
            method_std_values = []  # cette ligne stock les écart-types

            # Iterate through the subjects and look for the association
            for sub_metrics in subject_metrics:
                for k, v in sub_metrics.items():
                    if k == f"{name}_mean_{method}":
                        method_mean_values.append(v)
                    elif k == f"{name}_std_{method}":
                        method_std_values.append(v)  

            # Création d'une paire (mean, std) pour chaque méthode
            method_values_bar = (method_mean_values, method_std_values)
            metric_values_list_bar.append(method_values_bar)
        out_path = os.path.join(output_folder, f'{name}_{contrast}_bar_plot.png')
        save_bar(methods=methods_list, values=metric_values_list_bar, output_path=out_path, x_axis='Subjects', y_axis= f'{name} (pixels)')


    for metric_name in metric_name_only_list:
        metric_values_list = []
        for method_name in methods_list:
            method_values = []

        # Iterate through the subjects and look for the association
            for sub_metrics in subject_metrics:
                for k,v in sub_metrics.items():
                    if k == f"{metric_name}_{method_name}":
                        method_values.append(v)
            
            metric_values_list.append(method_values)

        out_path = os.path.join(output_folder, f'{metric_name}_{contrast}_violin_plot.png')
        save_violin(methods=methods_list, values=metric_values_list, output_path=out_path, x_axis='Methods', y_axis=f'{metric_name} (pixels)')
     

    
    # Save total Dice score DSC
    # DSC_hg = dict_total['DSC_hg']
    # DSC_sct = dict_total['DSC_sct']
    # DSC_spn = dict_total['DSC_spn']
    # out_path = os.path.join(output_folder,'labels_dice_score.png')
    # save_bar(x='Total',
    #          y1=DSC_hg,
    #          y2=DSC_sct,
    #          y3=DSC_spn,
    #          output_path=out_path,
    #          x_axis='Total', 
    #          y_axis='Labels accuracy using DSC', 
    #          label1 ='hourglass_network', 
    #          label2 ='sct_label_vertebrae', 
    #          label3 ='spinenetv2_label_vertebrae'
    #          )

def save_bar(methods, values, output_path, x_axis='Subjects', y_axis= 'Metric name (pixels)'):
    '''
    Create a bar graph
    :param methods: String list of the methods name
    :param values: List of tuples where each tuple contains (mean, std) corresponding to the methods name
    :param output_path: Path to output folder where figures will be stored
    :param x_axis: x-axis number of subject
    :param y_axis: y-axis name
    '''
    
    # set width of bar
    barWidth = 0.25
    
    #plt.subplots_adjust(bottom=0.2, top=0.9, left=0.05, right=0.95)
        
    # Set position of bar on X axis
    #br1 = np.arange(len(methods))

    mean_values, std_values = zip(*values)  # Séparation de la liste en deux listes distinctes avec les valeurs mean et std
    mean_values = list(mean_values)
    std_values = list(std_values) 
    br1 = np.arange(len(mean_values[0]))

    # Make the plot 
    plt.figure(figsize=(len(methods), 10))
    fig, ax = plt.subplots()  

    # Create the bar plots for each method
    for i, method in enumerate(methods):
        if any(value != 0 and value != 1 for value in mean_values[i]):              
            # Remove empty values initialized previously to 0 or 1 depending on the metric
            ax.bar(br1 + i * barWidth, mean_values[i], yerr=std_values[i], width=barWidth, label=method)
                    
    #ax.bar(br1, mean_values, yerr=std_values, align='center', color='b', width = barWidth, edgecolor ='grey')
    plt.title(f"bar plot of {y_axis}" , fontweight ='bold', fontsize = 20)


 
    # Create axis and adding Xticks
    plt.xlabel(x_axis, fontweight ='bold', fontsize = 15)
    plt.ylabel(y_axis, fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth * 1.5 for r in br1], [f"{i + 1}" for i in range(len(mean_values[0]))])
    #plt.xticks([r for r in range(len(methods))], methods)
    
    # Save plot
    plt.legend()
    plt.savefig(output_path)



"""def save_bar(methods, mean, std, output_path, x_axis='Subjects', y_axis='L2_error (pixels)'):
    '''
    Create a bar graph
    :param methods: String list of the methods name
    :param mean: List of the mean corresponding to the methods name
    :param str: List of the str corresponding to the methods name
    :param output_path: Path to output folder where figures will be stored
    :param x_axis: x-axis name
    :param y_axis: y-axis name
    '''
    
    # set width of bar
    barWidth = 0.25
    plt.figure(figsize =(len(methods), 10))
    #plt.subplots_adjust(bottom=0.2, top=0.9, left=0.05, right=0.95)
        
    # Set position of bar on X axis
    br1 = np.arange(len(methods))
    
    # Make the plot 
    fig, ax = plt.subplots()      
    ax.bar(br1, mean, yerr=std, align='center', color='b', width = barWidth, edgecolor ='grey')
    plt.title(y_axis, fontweight ='bold', fontsize = 20)
    
    # Create axis and adding Xticks
    plt.xlabel(x_axis, fontweight ='bold', fontsize = 15)
    plt.ylabel(y_axis, fontweight ='bold', fontsize = 15)
    plt.xticks([r for r in range(len(methods))], methods)
    
    # Save plot
    plt.legend()
    plt.savefig(output_path)
"""

def save_violin(methods, values, output_path, x_axis='Methods', y_axis='Metric name'):
    '''
    Create a bar graph
    :param methods: String list of the methods name
    :param values: Values associated with the methods
    :param output_path: Path to output folder where figures will be stored
    :param x_axis: x-axis name
    :param y_axis: y-axis name
    '''
    # set width of bar
    # Set position of bar on X axis
    result_dict = {'methods' : [], 'values' : []}
    """for i, method in enumerate(methods):
        if len(values[i]) > 0:
            result_dict['values'] += values[i]
            for j in range(len(values[i])):
                result_dict['methods'] += [method]"""

    for i, method in enumerate(methods):
        if any(value != 0 and value != 1 for value in values[i]):
            # Remove empty values initialized previously to 0 or 1 depending on the metric
            result_dict['values'] += values[i]
            #result_dict['values'].extend(values[i])
            #result_dict['methods'].extend([method] * len(values[i]))
            for j in range(len(values[i])):
                result_dict['methods'] += [method]

    result_df = pd.DataFrame(data=result_dict)

    # Make the plot 
    plt.figure()
    sns.violinplot(x = "methods", y = "values", data= result_df)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'{y_axis} violin plot')
  # Rotation des étiquettes de l'axe des x
    plt.xticks(rotation=45)

    # Ajustement des marges pour éviter que les noms soient coupés
    plt.subplots_adjust(bottom=0.20)
    
    # Save plot
    plt.savefig(output_path)


def calculate_auc(tpr, fpr):
    #Calculate AUC (Area Under the Curve)
    sorted_indices = np.argsort(fpr)
    sorted_tpr = np.array(tpr)[sorted_indices]
    sorted_fpr = np.array(fpr)[sorted_indices]
    auc_value = 0.0
    for i in range(1, len(sorted_fpr)):
        auc_value += (sorted_fpr[i] - sorted_fpr[i-1]) * (sorted_tpr[i] + sorted_tpr[i-1]) / 2.0
    return auc_value

def save_ROC_curves(methods, TPR_list, FPR_list, output_path, x_axis='True Positive Rate (TPR)', y_axis='False Positive Rate (FPR)'):
    plt.figure()

    for method in enumerate(methods):
        #tpr = TPR_list[i]
        #fpr = FPR_list[i]
        # Ignore les points où TPR et FPR sont tous deux à 0 ou à 1
        #if (tpr != 0 or fpr != 0) and (tpr != 1 or fpr != 1):
        #    print(f"Method: {method}, TPR: {tpr}, FPR: {fpr}")
        #    tpr =[tpr]
        #    fpr=[fpr]
        plt.plot(FPR_list, TPR_list, label=f'{method} (AUC = {calculate_auc(TPR_list, FPR_list):.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.title('ROC Curve')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.savefig(output_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute metrics on sct and hourglass disc estimation')
    
    parser.add_argument('-txt', '--input-txt-file', type=str, metavar='<file>', required=True,
                        help='Path to txt file generated using src/bcm/run/extract_disc_cords.py Example: ~/<your_dataset>/vertebral_data (Required)')
    parser.add_argument('--config-data', type=str, metavar='<folder>', default='',
                        help='Config JSON file where every label/image used for TESTING has its path specified ~/<your_path>/config_data.json (Required)')
    
    parser.add_argument('-o', '--output-folder', type=str, metavar='<folder>', default='results',
                        help='Output folder where created graphs and images will be stored (default="results")') 
    parser.add_argument('-csv', '--create-csv', type=bool, default=True,
                        help='If "True" generate a csv file with the computed metrics within the txt file folder (default=True)') 
    parser.add_argument('--gt-exists', type=bool, default=True,
                        help='If "True" compute metrics only if GT exists (default=True)') 
    parser.add_argument('--computed-methods', 
                        default=['sct_discs_coords', 
                                 'spinenet_coords',
                                 'nnunet_coords_101',
                                 'nnunet_coords_102',
                                 'nnunet_coords_200',
                                 'nnunet_coords_201',
                                 'hourglass_t1_coords', 
                                 'hourglass_t2_coords', 
                                 'hourglass_t1_t2_coords'],
                        help='Methods on which metrics will be computed'
                        '["sct_discs_coords", "spinenet_coords","nnunet_coords_101","nnunet_coords_102", "nnunet_coords_200", "nnunet_coords_201","hourglass_t1_coords", "hourglass_t2_coords", "hourglass_t1_t2_coords"]')
    
    
    compare_methods(parser.parse_args())

    print('All metrics have been computed')