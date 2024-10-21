import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


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


def save_violin(methods, values, hue=[], output_path='test.png', x_axis='Methods', y_axis='Metric name'):
    '''
    Create a bar graph
    :param methods: String list of the methods name
    :param values: Values associated with the methods
    :param hue: Class associated with the methods
    :param output_path: Path to output folder where figures will be stored
    :param x_axis: x-axis name
    :param y_axis: y-axis name
    '''
    # set width of bar
    # Set position of bar on X axis
    result_dict = {'methods' : [], 'values' : [], 'Class' : []}
    for i, method in enumerate(methods):
        if len(values[i]) > 0:
            result_dict['values'] += values[i]
            for j in range(len(values[i])):
                result_dict['methods'] += [method]
                result_dict['Class'] += [hue[i]] if hue else [method]


    result_df = pd.DataFrame(data=result_dict)
    sns.set_theme(style="darkgrid")

    # Make the plot 
    plt.figure(figsize=(len(methods)//2, 8))
    ax = sns.violinplot(x="methods", y="values", hue="Class", data=result_df, cut=0, width=0.9, bw_method=1)
    if len(methods) > 20:
        xticks = [f'{method}' if i%2==0 else f'\n{method}' for i, method in enumerate(methods)] # Shift label up and down
        plt.xticks(list(range(len(xticks))), xticks, fontsize=14)
    else:
        plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'{y_axis} violin plot')
    plt.xlabel(x_axis, fontsize = 20)
    plt.ylabel(y_axis, fontsize = 20)
    plt.title(y_axis, fontsize = 25)

    # Extract the colors used for the hue from the plot
    handles, labels = ax.get_legend_handles_labels()
    legend_colors = [handle.get_facecolor() for handle in handles]

    # Match the x-tick labels' colors to the corresponding hue colors
    for i, label in enumerate(ax.get_xticklabels()):
        method = methods[i]  # Corresponding method
        # Retrieve the associated class for the current method
        associated_class = result_df[result_df["methods"] == method]["Class"].values[0]
        class_idx = list(result_df["Class"].unique()).index(associated_class)  # Find the index of the class
        label.set_color(legend_colors[class_idx])  # Set the color based on the hue class

    # Increase the font size of the legend (hue label)
    plt.legend(title='Class', title_fontsize=18, fontsize=18, loc='best')

    # Adjust spacing between the plot and labels
    plt.tight_layout()
    
    # Save plot
    print(f'Figure saved under {output_path}')
    plt.savefig(output_path)
