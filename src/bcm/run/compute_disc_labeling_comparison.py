import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd
import json

from bcm.utils.utils import SCT_CONTRAST, edit_metric_csv, fetch_img_and_seg_paths, fetch_subject_and_session
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



    ################################################################################################################################################################################################
    #Morane --- method - resul qui est le dictionnaire qui compte toutes les metrics de chaque methode -- à modifier quand on ajoute une methode
    # Morane -- contraste = T1w/T2w /// lignes d'en haut possible erreur
    ################################################################################################################################################################################################



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
            if args.config_data:
                sub_name = f'{subject}_{contrast}'
                for img_path in img_paths:
                    if sub_name in img_path:
                        img_3D = Image(img_path).change_orientation('RSP').data
                        shape = img_3D.shape
                        img_2D = img_3D[shape[0]//2, :, :]
                        #if pred_discs_list.size != 0: # Check if not empty
                            #visualize_discs(input_img=img_2D, coords_list=pred_discs_list, out_path=os.path.join(output_folder, f'{sub_name}_{method}.png'))
            
    
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
            else:
                methods_plot.append(method.split('_coords')[0]) # Remove '_coords' suffix
        save_graphs(output_folder, methods_results[contrast], methods_plot, contrast)

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
    metrics_values = np.array([list(sub_metrics.values()) for sub_metrics in subject_metrics])
    
    ###############
    for metric_name in metrics_name:
        # Plot violin graph for each metric
        metric_data = [metrics_values[:, np.where(metrics_name == metric_name)[0][0]] for method in methods_list]
        
        # Check if mean and std are available for the metric
        if f'{metric_name}_mean' in dict_total and f'{metric_name}_std' in dict_total:
            metric_mean = [dict_total[f'{metric_name}_mean_{method}'] for method in methods_list]
            metric_std = [dict_total[f'{metric_name}_std_{method}'] for method in methods_list]
        else:
            # If not available, use empty lists
            metric_mean = []
            metric_std = []
        
        out_path = os.path.join(output_folder, f'{metric_name}_{contrast}.png')
        save_violin(methods=methods_list, values=metric_data, output_path=out_path, x_axis='Methods', y_axis=f'{metric_name} (pixels)')
    ###############
    # Plot total graph
    # L2 error
    #l2_error = [metrics_values[:,np.where(metrics_name == f'l2_mean_{method}')[0][0]] for method in methods_list]
    #l2_mean = [dict_total[f'l2_mean_{method}'] for method in methods_list]
    #l2_std = [dict_total[f'l2_std_{method}'] for method in methods_list]
    #out_path = os.path.join(output_folder,f'l2_error_{contrast}.png')
    #save_bar(methods=methods_list, mean=l2_mean, std=l2_std, output_path=out_path, x_axis='Methods', y_axis='L2_error (pixels)')
    #save_violin(methods=methods_list, values=l2_error, output_path=out_path, x_axis='Methods', y_axis='L2_error (pixels)')


    # # Save total Dice score DSC
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

def save_violin(methods, values, output_path, x_axis='Subjects', y_axis='Metric name'):
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
    result_dict = {}
    for i, method in enumerate(methods):
        result_dict[method]=values[i]
    result_df = pd.DataFrame(data=result_dict)

    # Make the plot 
    plot = sns.violinplot(data=result_df)  
    plot.set(xlabel='methods', ylabel='metric name (pixels)')
    plot.set(title='Position error (pixels)')
    #plot.set(xlabel='methods', ylabel='L2 error (pixels)')
    #plot.set(title='Position error (pixels)')
    #plot.set_ylim(-10,60)
    # Create axis and adding Xticks
    
    # Save plot
    plot.figure.savefig(output_path)

#def hello():
#    print('hello')

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
                                 'hourglass_t1_coords', 
                                 'hourglass_t2_coords', 
                                 'hourglass_t1_t2_coords'],
                        help='Methods on which metrics will be computed'
                        '["sct_discs_coords", "spinenet_coords", "hourglass_t1_coords", "hourglass_t2_coords", "hourglass_t1_t2_coords"]')
    
    
    compare_methods(parser.parse_args())

    print('All metrics have been computed')