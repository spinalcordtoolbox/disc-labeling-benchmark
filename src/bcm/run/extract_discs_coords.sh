#!/bin/bash

# Check for -h option
if [ "$1" = "-h" ]; then
    echo "This script is designed to compute disc labeling technics on an input dataset "
    echo "The data is expected to be gathered using the src/bcm/utils/gather_data.py script. "
    echo "The main steps of this script include: "
    echo ""
    echo "1. Creating and initializing a text file were all the coordinates for each method"
    echo " where all the coordinates of the discs will be gathered "
    echo "2. Running the scripts of the wanted methods available in this benchmark "
    echo "3. Computing the metrics and creating the graphs and images to compare the different "
    echo "approaches"
    echo ""
    echo "The script takes several command-line arguments for customization:"
    echo "--datapath: path to the input dataset"

    echo "Data organization - using the script gather_data.py:"
    echo " data"
    echo " ├── sub-errsm37"
    echo " │   ├── sub-errsm37_T2w_labels-disc-manual.nii.gz"
    echo " │   ├── sub-errsm37_T2w.nii.gz"
    echo " │   ├── sub-errsm37_T1w_labels-disc-manual.nii.gz"
    echo " │   └── sub-errsm37_T1w.nii.gz"
    echo " └── sub-errsm38"
    echo "     ├── sub-errsm38_T1w_seg-manual.nii.gz"
    echo "     └── sub-errsm38_T1w.nii.gz"

    exit 0
fi

# BASH SETTINGS
# ======================================================================================================================

# Uncomment for full verbose
# set -v

# Immediately exit if error
set -e

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# GET PARAMS
# ======================================================================================================================

# SET DEFAULT VALUES FOR PARAMETERS.
# ----------------------------------------------------------------------------------------------------------------------
# General parameters
DATA_DIR=""
CONTRAST=""
OUTPUT_DIR="results/files/"
OUTPUT_TXT=""
SUFFIX_IMG="_acq-sagittal"
SUFFIX_LABEL_DISC="_labels-manual"
SUFFIX_SEG="_seg"
VERBOSE=1

# Hourglass config file
CONFIG_HG=""


# Get command-line parameters to override default values.
# ----------------------------------------------------------------------------------------------------------------------
params="$(getopt -o d:c:f:ov -l datapath:,contrast:,file:,out:,verbose --name "$0" -- "$@")"
eval set -- "$params"

while true
do
    case "$1" in
        -d|--datapath)
            DATA_DIR="$2"
            shift 2
            ;;
        -c|--contrast)
            CONTRAST="$2"
            shift 2
            ;;
        -f|--file)
            CONFIG_HG+=" $2"
            shift 2
            ;;
        -o|--out)
            OUTPUT_TXT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Not implemented: $1"
            exit 1
            ;;
    esac
done

# Get full path for all parameters.
DATA_DIR=$(realpath "${DATA_DIR}")
OUTPUT_DIR=$(realpath "${OUTPUT_DIR}")

# Define default value OUTPUT_TXT
if [[ -z "$OUTPUT_TXT" ]] ; then
    OUTPUT_TXT="${OUTPUT_DIR}/$(basename -- $DATA_DIR)_${CONTRAST}_discs_coords.txt"
fi

# Print the parameters if VERBOSE is enabled.
# ----------------------------------------------------------------------------------------------------------------------
if [[ ${VERBOSE} == 1 ]]; then
    echo ""
    echo "Running with the following parameters:"
    echo "DATA_DIR=${DATA_DIR}"
    echo "CONTRAST=${CONTRAST}"
    echo "CONFIG_HG=${CONFIG_HG}"
    echo "OUTPUT_TXT=${OUTPUT_TXT}"
    echo "VERBOSE=${VERBOSE}"
    echo ""
fi

# Validate the given parameters.
# ----------------------------------------------------------------------------------------------------------------------

# Ensure the data directory exists.
if [[ ! -d ${DATA_DIR} ]]; then
    echo "Folder not found ${DATA_DIR}"
    exit 1
fi

# Ensure the output directory exists, creating it if necessary.
mkdir -p ${OUTPUT_DIR}
if [[ ! -d ${OUTPUT_DIR} ]]; then
    echo "Folder not found ${OUTPUT_DIR}"
    exit 1
fi

# SCRIPT STARTS HERE
# ======================================================================================================================

# Mandatory args
args=( --datapath "$DATA_DIR" --contrast "$CONTRAST" --out-txt-file "$OUTPUT_TXT")

# Methods args
args_m=${args[*]}
if [ ! -z "$SUFFIX_IMG" ]; then args_m+=( --suffix-img "$SUFFIX_IMG" );fi
if [ ! -z "$SUFFIX_LABEL_DISC" ]; then args_m+=( --suffix-label-disc "$SUFFIX_LABEL_DISC" );fi
if [ ! -z "$SUFFIX_SEG" ]; then args_m+=( --suffix-seg "$SUFFIX_SEG" );fi

## Activate env HOURGLASS + SCT
source /usr/local/miniforge3/etc/profile.d/conda.sh
conda activate sct_hg_env

# Init text file
python src/bcm/utils/init_txt_file.py ${args[@]}

# Add ground truth discs coordinates
python src/bcm/methods/add_gt_coordinates.py ${args_m[@]}

# Test sct_label_vertebrae
python src/bcm/methods/test_sct_label_vertebrae.py ${args_m[@]}

# Test Hourglass Network
# Add config file parameter for hourglass
for file in $CONFIG_HG
    do
        file=$(realpath "${file}")
        args_hg=${args_m[*]}
        args_hg+=" --config-hg $file " # why quotes ?? and not parentheses
        python src/bcm/methods/test_hourglass_network.py ${args_hg[@]}
    done

## Deactivate env
conda deactivate

# Test Spinenet Network with spinenet-venv
../spinenet-venv/bin/python src/bcm/methods/test_spinenet_network.py "${args_m[@]}"

echo "All the methods have been computed"
