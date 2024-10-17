import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd
import json

from bcm.utils.utils import edit_metric_csv, fetch_bcm_paths, visualize_discs
from bcm.utils.image import Image


def compare_methods(args):
    if args.config_data:
        config_data = json.load(open(args.config_data, "r"))

        # Get image and segmentation paths
        img_paths, _, _ = fetch_bcm_paths(config_data)

    txt_file_path = args.input_txt_file
    dataset = os.path.basename(txt_file_path).split('_')[0]
    output_folder = os.path.join(args.output_folder, f'out_{dataset}')

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load disc_coords txt file
    with open(txt_file_path,"r") as f:  # Checking already processed subjects from txt file
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
        split_lines[0][-1] = split_lines[0][-1].replace('\n','')
    
    # Extract methods list
    num_disc_idx = split_lines[0].index('num_disc')
    computed_methods = split_lines[0][num_disc_idx+1:]
    if not 'gt_coords' in computed_methods:
        raise ValueError('Ground truth labels need to be present for comparison')
    else:
        computed_methods.remove('gt_coords')

    # Extract processed subjects --> subjects with a ground truth
    processed_subjects_dict = dict()
    for line in split_lines[1:]:
        gt_condition = (line[split_lines[0].index('gt_coords')]!='None') # Process subject only if ground truth are available
        if (line[split_lines[0].index('subject_name')] not in processed_subjects_dict.keys()) and gt_condition:
            processed_subjects_dict[line[split_lines[0].index('subject_name')]] = [line[split_lines[0].index('contrast')]]
        elif (line[split_lines[0].index('subject_name')] in processed_subjects_dict.keys()) and gt_condition:
            if (line[split_lines[0].index('contrast')] not in processed_subjects_dict[line[split_lines[0].index('subject_name')]]):
                processed_subjects_dict[line[split_lines[0].index('subject_name')]].append(line[split_lines[0].index('contrast')])


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

            # Visualize discs on image
            if args.config_data:
                sub_name = f'{subject}_{contrast}'
                for img_path in img_paths:
                    if sub_name in img_path:
                        img_3D = Image(img_path).change_orientation('RSP').data
                        shape = img_3D.shape
                        img_2D = img_3D[shape[0]//2, :, :]
                        if pred_discs_list.size != 0: # Check if not empty
                            visualize_discs(input_img=img_2D, coords_list=pred_discs_list, out_path=os.path.join(output_folder, f'{sub_name}_{method}.png'))
    
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
        save_graphs(output_folder, methods_results[contrast], computed_methods, contrast)


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
    position_tot = [i for i, name in enumerate(metrics_name) if "tot_" in name]

    # Remove values from metrics_name at the found positions
    metrics_name = [name for i, name in enumerate(metrics_name) if i not in position_tot]

    # metrics_name contains only the remaining names
    metrics_values = np.array([list(sub_metrics.values()) for sub_metrics in subject_metrics])
    # 2D Numpy array, where each row corresponds to a subject, and each column corresponds to a metric.

    # Delete columns corresponding to a name with "tot_" in each row of the array
    metrics_values = np.delete(metrics_values, position_tot, axis=1)

    metrics_mean_list=[]
    metrics_std_list=[]
    metric_name_only_list = []
    
    for metric_name in metrics_name:
        # Find the last "_mean" in the metric name
        last_mean_index = metric_name.rfind("_mean")
        # Find the last "_std" in the metric name
        last_std_index = metric_name.rfind("_std")
        metric_name_parts = metric_name.split("_", 1)
        metric_std = ''
        metric_mean = ''

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


    for metric_name in metric_name_only_list:
        metric_values_list = []
        for method_name in methods_list:
            method_values = []

            # Iterate through the subjects and look for the association
            for sub_metrics in subject_metrics:
                for k,v in sub_metrics.items():
                    if v != -1:
                        if k == f"{metric_name}_{method_name}":
                            method_values.append(v)
            
            metric_values_list.append(method_values)
        
        out_path = os.path.join(output_folder, f'{metric_name}_{contrast}_violin_plot.png')
        if metric_name.startswith('z') or metric_name.startswith('l2'):
            metric_name = f'{metric_name} (pixels)'
        print(f'Saving violin plot for metric {metric_name}')
        save_violin(methods=methods_list, values=metric_values_list, output_path=out_path, x_axis='Methods', y_axis=metric_name)
    

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
    for i, method in enumerate(methods):
        if len(values[i]) > 0:
            result_dict['values'] += values[i]
            for j in range(len(values[i])):
                result_dict['methods'] += [method]


    result_df = pd.DataFrame(data=result_dict)
    sns.set_theme(style="darkgrid")

    # Make the plot 
    plt.figure(figsize=(13, 8))
    sns.violinplot(x="methods", y="values", hue="methods", data=result_df, cut=0, width=0.7, bw_method=.2)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'{y_axis} violin plot')
    plt.xlabel(x_axis, fontsize = 20)
    plt.ylabel(y_axis, fontsize = 20)
    plt.title(y_axis, fontsize = 25)
    
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
    
    
    compare_methods(parser.parse_args())

    print('All metrics have been computed')