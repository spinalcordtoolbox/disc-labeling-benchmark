#===================================================
## Authors: 
# - Reza Azad (rezazad68@gmail.com)
# - Nathan Molinier (nathan.molinier@gmail.com)
#===================================================

from shutil import copyfile
import os
import argparse


def main(args):
    '''
    Gather only relevant data for the hourglassnetwork.
    '''
    data_dir = args.datapath # The data need to use BIDS convention
    label_suffix = args.suffix_label
    destination_path = args.output_folder
    
    ADD =  data_dir + "derivatives/labels/"
    ADD2 = data_dir

    list_dir = os.listdir(ADD)
    Total = 0
    for idx in range (len(list_dir)):   
        ## Copy the T1 disc label   
        src1 = ADD + list_dir[idx] + '/anat/'+list_dir[idx] + '_T1w' + label_suffix + '.nii.gz'
        dst1 = list_dir[idx] + '/'         +list_dir[idx] + '_T1w' + label_suffix +'.nii.gz'
        
        ## Copy the T2 disc label   
        src2 = ADD + list_dir[idx] + '/anat/'+list_dir[idx] + '_T2w' + label_suffix + '.nii.gz'
        dst2 = list_dir[idx] + '/'         +list_dir[idx] + '_T2w' + label_suffix + '.nii.gz'    
        
        ## Copy the T1 file   
        src3 = ADD2 + list_dir[idx] + '/anat/'+list_dir[idx] + '_T1w.nii.gz'
        dst3 = list_dir[idx] + '/'          +list_dir[idx] + '_T1w.nii.gz'    
        
        ## Copy the T2 file   
        src4 = ADD2 + list_dir[idx] + '/anat/'+list_dir[idx] + '_T2w.nii.gz'
        dst4 = list_dir[idx] + '/'          +list_dir[idx] + '_T2w.nii.gz'            
        
        if os.path.exists(src1):
            if not os.path.exists(destination_path+list_dir[idx]):
                os.makedirs(os.path.join(destination_path, list_dir[idx]))
            copyfile(src1, os.path.join(destination_path, dst1))
            copyfile(src2, os.path.join(destination_path, dst2))
            copyfile(src3, os.path.join(destination_path, dst3))
            copyfile(src4, os.path.join(destination_path, dst4))
            Total += 1

    print(f'Total number of {Total} subject selected')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gather image files and vertebral labels in a new directory')
    
    ## Parameters
    parser.add_argument('--datapath', type=str, required=True,
                        help='Path to BIDS data')
    parser.add_argument('-o', '--output-folder', type=str, required=True,
                        help='Path out to output folder')
    parser.add_argument('--suffix-label', type=str, default='_labels-disc-manual',
                        help='Specify label suffix (default= "_labels-disc-manual")') 
    
    main(parser.parse_args())