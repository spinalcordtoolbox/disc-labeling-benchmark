import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
from functools import partial
import argparse
from pathlib import Path
import os
import numpy as np
import json
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

from bcm.utils.metrics import compute_dsc
from bcm.utils.image import Image
from bcm.utils.plot import save_boxplot

def main(args):
    # Load variables
    gt_dir = args.gt_dir
    gt_map_path = args.gt_map
    pred_dir = args.pred_dir
    pred_map_path = args.pred_map
    workers = args.max_workers

    # Check if paths exist
    for path in [gt_dir, gt_map_path, pred_dir, pred_map_path]:
        if not os.path.exists(path):
            raise ValueError(f'Error: {path} does not exist')
    
    # Check if gt folder and pred folder have the same number of images
    if len(os.listdir(gt_dir)) != len(os.listdir(pred_dir)):
        raise ValueError('The ground truth and prediction folders have a different number of images')
    
    # Load JSON files and create a dictionary
    with open(gt_map_path, "r") as file:
        gt_map = json.load(file)
    
    with open(pred_map_path, "r") as file:
        pred_map = json.load(file)
    
    # Use multiprocessing for evaluation
    subs_matrix = evaluate_seg_mp(gt_folder=gt_dir, pred_folder=pred_dir, gt_map=gt_map, pred_map=pred_map, workers=workers)

    # Extract diagonal
    diag_dice = np.array([np.diagonal(mat) for mat in subs_matrix])
    value_list = [list(diag_dice[:,i]) for i in range(len(gt_map.keys()))]

    # Create multi class dice violin plot
    hue = ['Intervertebral discs' if '-' in name else 'Vertebrae' for name in list(gt_map.keys())]
    save_boxplot(list(gt_map.keys()), value_list, hue=hue, output_path='multi_class_dice.png', x_axis='Class', y_axis='Dice')

    # Set predictions to 1 if DICE above threshold
    # threshold = 0.8
    # subs_matrix_threshold = [np.where(m>threshold,1,0) for m in subs_matrix]

    # # Combine prediction in a summary table
    # sum_matrix = np.sum(subs_matrix_threshold, axis=0)

    # # Plot matrix
    # mpl.rc('image', cmap='nipy_spectral_r')
    # plt.figure()
    # plt.xticks(range(len(gt_map.keys())), gt_map.keys())
    # plt.yticks(range(len(gt_map.keys())), gt_map.keys())
    # plt.xlabel('ground truth')
    # plt.ylabel('prediction')
    # plt.imshow(sum_matrix)
    # plt.colorbar()
    # plt.savefig('labeling_evaluation.png')



def evaluate_seg_mp(
        gt_folder,
        pred_folder,
        gt_map, 
        pred_map,
        workers=mp.cpu_count(),
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    filenames = os.listdir(gt_folder)
    gt_paths = [Path(gt_folder) / name for name in filenames]
    pred_paths = [Path(pred_folder) / name for name in filenames]

    with mp.Pool(processes=workers) as pool:
        args = zip(gt_paths, pred_paths)
        mats = pool.starmap(
            partial(
                    evaluate_seg,
                    gt_map=gt_map, 
                    pred_map=pred_map,
            ),
            tqdm.tqdm(args, total=len(gt_paths))
        )
    return mats

def worker(x, y, num):
    return x + y + num

def evaluate_seg(
        gt_path,
        pred_path,
        gt_map, 
        pred_map
    ):

    # Extract labels from ground truth
    labels = gt_map.keys()

    # Init output matrix
    out_matrix = np.zeros((len(labels), len(labels)))

    # Load ground truth and prediction
    gt_seg = Image(str(gt_path)).change_orientation('RSP').data
    pred_seg = Image(str(pred_path)).change_orientation('RSP').data

    if gt_seg.shape == pred_seg.shape:
        # Loop on the labels
        for i, gt_label in enumerate(labels):
            gt_mask = np.isin(gt_seg, gt_map[gt_label]).astype(int)
            pred_label = gt_label
            #for j, pred_label in enumerate(labels):
            # Extract segmented structure from ground truth
            pred_mask = np.isin(pred_seg, pred_map[pred_label]).astype(int)
            dsc = compute_dsc(gt_mask, pred_mask)
            out_matrix[i,i]=dsc
    return out_matrix


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute the evaluation of segmentation tools against ground truth')

    ## Parameters
    # All mandatory parameters                         
    parser.add_argument('--gt-dir', type=str, metavar='<folder>', required=True,
                        help='Path to the folder containing the ground truth segmentations in Niftii format. The names of the files must match with the folder --pred-dir. (e.g. ~/<your_path>/<myfolder>) (Required)')
    parser.add_argument('--pred-dir', type=str, metavar='<folder>', required=True,
                        help='Path to the folder containing the predicted segmentations in Niftii format. The names of the files must match with the folder --gt-dir. (e.g. ~/<your_path>/<myfolder>) (Required)')
    parser.add_argument('--gt-map', type=str, metavar='<json>', required=True,
                        help='JSON file containing the GT mapping between the imaged structure and the corresponding integer value in the image ~/<your_path>/<myjson>.json (Required)')
    parser.add_argument('--pred-map', type=str, metavar='<json>', required=True,
                        help='JSON file containing the prediction mapping between the imaged structure and the corresponding integer value in the image ~/<your_path>/<myjson>.json (Required)')
    parser.add_argument('--max-workers', '-w', type=int, default=mp.cpu_count(), 
                        help='Max workers to run in parallel processes, defaults to multiprocessing.cpu_count().')
    
    args = parser.parse_args()
    
    # Run labeling evaluation
    main(parser.parse_args())

    print('The labeling evaluation was computed')