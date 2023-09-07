#!/bin/bash

# Check for -h option
if [ "$1" = "-h" ]; then
    echo "This script is designed to compute disc labeling technics on an input dataset "
    echo "The data path used for testing should be gathered into a JSON file using the src/bcm/utils/init_data_config.py script. "
    echo "The script includes 2 main steps: "
    echo ""
    echo "1. Creating a text file where all the coordinates of every discs for each method will be gathered"
    echo ""
    echo "2. Computing the metrics and creating the graphs and images to compare the different approaches"
    echo ""
    echo "The script takes several command-line arguments for customization:"
    echo ""

    echo "Path specified need to follow BIDS compatibility:"
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
CONFIG_DATA=""
OUTPUT_DIR="results/"
OUTPUT_TXT=""
SUFFIX_SEG="_seg-manual"
VERBOSE=1

# Hourglass config file
CONFIG_HG=""


# Get command-line parameters to override default values.
# ----------------------------------------------------------------------------------------------------------------------
params="$(getopt -o d:f:ov -l data:,contrast:,file:,out:,verbose --name "$0" -- "$@")"
eval set -- "$params"

while true
do
    case "$1" in
        -d|--data)
            CONFIG_DATA="$2"
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
CONFIG_DATA=$(realpath "${CONFIG_DATA}")
OUTPUT_DIR=$(realpath "${OUTPUT_DIR}")

# Define default value OUTPUT_TXT
if [[ -z "$OUTPUT_TXT" ]] ; then
    OUTPUT_TXT="${OUTPUT_DIR}/files/${CONTRAST}_discs_coords.txt"
fi

# Print the parameters if VERBOSE is enabled.
# ----------------------------------------------------------------------------------------------------------------------
if [[ ${VERBOSE} == 1 ]]; then
    echo ""
    echo "Running with the following parameters:"
    echo "CONFIG_DATA=${CONFIG_DATA}"
    echo "CONFIG_HG=${CONFIG_HG}"
    echo "OUTPUT_TXT=${OUTPUT_TXT}"
    echo "VERBOSE=${VERBOSE}"
    echo ""
fi

# Validate the given parameters.
# ----------------------------------------------------------------------------------------------------------------------

# Ensure the data config exists.
if [[ ! -f ${CONFIG_DATA} ]]; then
    echo "File not found ${CONFIG_DATA}"
    exit 1
fi

# Ensure the HG config exists.
if [[ ! -f ${CONFIG_HG} ]]; then
    echo "File not found ${CONFIG_HG}"
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
args=( --config-data "$CONFIG_DATA" --out-txt-file "$OUTPUT_TXT" --suffix-seg "$SUFFIX_SEG" --seg-folder "$OUTPUT_DIR" )

## Activate env HOURGLASS + SCT
source /usr/local/miniforge3/etc/profile.d/conda.sh
conda activate hg_env

# Init text file
python src/bcm/utils/init_benchmark.py ${args[@]}

# Add ground truth discs coordinates
python src/bcm/methods/add_gt_coordinates.py ${args[@]}

# Test Hourglass Network
# Add config file parameter for hourglass
for file in $CONFIG_HG
    do
        file=$(realpath "${file}")
        args_hg=${args[*]}
        args_hg+=" --config-hg $file " # why quotes ?? and not parentheses
        python src/bcm/methods/test_hourglass_network.py ${args_hg[@]}
    done

## Change env
conda deactivate
conda activate sct_env

# Test sct_label_vertebrae
python src/bcm/methods/test_sct_label_vertebrae.py ${args[@]}

## Deactivate env
conda deactivate

# Test Spinenet Network with spinenet-venv
../spinenet-venv/bin/python src/bcm/methods/test_spinenet_network.py "${args[@]}"

echo "All the methods have been computed"
