# Intervertebral Discs Labeling Benchmark

This repository contains an evaluation and comparison of various automatic methods for intervertebral disc labeling. The goal of this project is to provide researchers and developers with a comprehensive overview of existing algorithms and techniques for accurately and efficiently labeling intervertebral discs in MRI images.

## Motivation

Accurate identification and labeling of intervertebral discs are crucial in medical imaging analysis, particularly in tasks related to spinal pathology assessment, surgery planning, and biomechanical modeling. However, manually labeling these discs is a time-consuming and labor-intensive process. Therefore, the development of automated methods to perform this task is of great importance, as it can significantly improve efficiency and consistency in clinical practice.

## Repository Structure

The repository is organized as follows:

- **Evaluation** (in development): This directory contains evaluation scripts and metrics to compare and quantify the performance of different labeling methods. It provides tools to assess accuracy, precision, recall, and other relevant measures.

- **Results** (in development): After running the evaluation scripts, the obtained results will be stored in this directory. It includes performance metrics, visualizations, and comparisons of the different methods.

## Getting Started

To get started with this repository, follow the steps below:

1. Clone the repository to your local machine using the command:
```Bash
git clone https://github.com/spinalcordtoolbox/disc-labeling-benchmark.git
```

2. Set up the required environment and dependencies. 

(in development) --> see https://github.com/spinalcordtoolbox/disc-labeling-benchmark/issues/2

3. Gather only the relevant data for comparison (The input dataset needs to be in [BIDS](https://bids.neuroimaging.io/) format): the `DATAPATH` corresponds to the path to the input BIDS compliant dataset and `VERTEBRAL_DATA` corresponds to the path to the output folder. The free multi-center spinegeneric dataset is available in https://github.com/spine-generic/data-multi-subject.
```Bash
python src/bcm/utils/gather_data.py --datapath DATAPATH -o VERTEBRAL_DATA --suffix-img SUFFIX_IMG --suffix-label SUFFIX_LABEL
```

4. Extract the coordinates of the discs for each image in the `VERTEBRAL_DATA` and create a `TXT_FILE` in results/
```Bash
python src/bcm/run/extract_disc_cords.py --datapath VERTEBRAL_DATA -c t2
```

5. Compute metrics and plot graphs for each methods based on the `TXT_FILE`. A `CSV_FILE` is also generated for more evaluation
```Bash
python src/bcm/run/compute_disc_labeling_comparison.py --datapath VERTEBRAL_DATA -txt results/files/spinegeneric_vert_T1w_hg15_discs_coords.txt -c t2
```

## Contributions and Feedback

Contributions to this repository are welcome. If you have developed a new method or have improvements to existing methods, please submit a pull request. Additionally, feedback and suggestions for improvement are highly appreciated. Feel free to open an issue to report bugs, propose new features, or ask questions.

## License

For more information regarding the license, please refere to the LICENSE file.
