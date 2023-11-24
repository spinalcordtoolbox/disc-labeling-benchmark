# Benchmark framework for Intervertebral Discs Labeling 

This repository contains an evaluation and comparison of various automatic methods for intervertebral disc labeling. The goal of this project is to provide researchers and developers with a comprehensive overview of existing algorithms and techniques for accurately and efficiently labeling intervertebral discs in MRI images.

## Motivation

Accurate identification and labeling of intervertebral discs are crucial in medical imaging analysis, particularly in tasks related to spinal pathology assessment, surgery planning, and biomechanical modeling. However, manually labeling these discs is a time-consuming and labor-intensive process. Therefore, the development of automated methods to perform this task is of great importance, as it can significantly improve efficiency and consistency in clinical practice.

## Repository Structure

The repository is organized as follows:

- **Results** : After running the evaluation scripts, the obtained results will be stored in this directory. It includes performance metrics, visualizations, and comparisons of the different methods.

## Getting Started

To get started with this repository, follow the steps below:

1. Clone the repository to your local machine using the command:
```Bash
git clone https://github.com/spinalcordtoolbox/disc-labeling-benchmark.git
```

2. Set up the required environments and dependencies. This repository contains several methods with independant environments, please install
each environment as follows:

<details>
<summary>Hourglass network</summary>
<br>
First, create a new virtual environment using python3.8 and activate it:
<details>
<summary>Conda</summary>
  
```Bash
conda create -n HG_env python=3.8
conda activate HG_env
```

</details>
<details>
<summary>Venv</summary>
Be sure to run python 3.8
  
```Bash
python -m venv HG_env
source HG_env/bin/activate
```

</details>
Then, install the packages by running these commands:

```Bash
git clone https://github.com/spinalcordtoolbox/disc-labeling-hourglass.git
cd disc-labeling-hourglass
pip install -r requirements.txt
pip install -e .
cd ..
```

</details>

<details>
<summary>Spinenet network</summary>
<br>
First, create a new virtual environment activate it, you can also follow spinenet [installation](https://github.com/rwindsor1/SpineNet#install-enviroments/):

```Bash
python -m venv spinenet-venv
source spinenet-venv/bin/activate
```

Then, install the packages by running these commands:
```Bash
git clone https://github.com/rwindsor1/SpineNet.git
cd SpineNet
pip install -r requirements.txt
cd ..
```

Before running, add the root directory to your PYTHONPATH:

```Bash
export PYTHONPATH=$PYTHONPATH:/path/to/SpineNet
```

Finally, download spinenet's weight using this command

```Bash
spinenet.download_weights(verbose=True)
```
</details>

<details>
<summary>Spinalcordtoolbox installation</summary>
<br>
In this benchmark, few features including the function sct_label_vertebrae from the [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox/) are needed. Instructions regarding the installation follows:
  
```Bash
git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox.git
cd spinalcordtoolbox
./install_sct
``` 
</details>

3. Gather only the relevant data for comparison in a json file `CONFIG_DATA_JSON` (The input data needs to be stored in a [BIDS](https://bids.neuroimaging.io/) compliant dataset): all the `labels`' path need to be stored in this `json` file before running any script in the benchmark. The different steps are described [here](https://github.com/spinalcordtoolbox/disc-labeling-hourglass/issues/25#issuecomment-1695818382).
> Note : the script `init_data_config.py` is also available within this repository in `src/bcm/utils/init_data_config.py`

4. Extract the coordinates of the discs for each image in the `CONFIG_DATA_JSON` and create a `TXT_FILE` in results/
   
```Bash
src/bcm/run/extract_discs_coords.sh --data CONFIG_DATA_JSON --file CONFIG_HG
```

5. Compute metrics and plot graphs for each methods based on the `TXT_FILE`. A `CSV_FILE` is also generated for more evaluation
   
```Bash
python src/bcm/run/compute_disc_labeling_comparison.py --config-data CONFIG_DATA_JSON -txt results/files/spinegeneric_vert_T1w_hg15_discs_coords.txt
```
## Methods

- Hourglass
- sct_label_vertebrae
- spinenet
- nnunet101: discs segmentations + classes
- nnunet102: discs segmentations
- nnunet200: discs coordinates + classes
- nnunet201: discs coordinates

## See also

NeuroPoly disc labeling implementations:
- Hourglass approach: https://github.com/spinalcordtoolbox/disc-labeling-hourglass
- nnU-Net approach: https://github.com/spinalcordtoolbox/disc-labeling-nnunet

## Contributions and Feedback

Contributions to this repository are welcome. If you have developed a new method or have improvements to existing methods, please submit a pull request. Additionally, feedback and suggestions for improvement are highly appreciated. Feel free to open an issue to report bugs, propose new features, or ask questions.

## License

For more information regarding the license, please refere to the LICENSE file.
