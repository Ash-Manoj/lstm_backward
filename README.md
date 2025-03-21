# LSTM Backward

This repository contains code and data for replicating the results from Manoj J et al. [2024](https://doi.org/10.5194/hess-2024-375). The code's logic is heavily based on [Hy2DL](https://github.com/KIT-HYD/Hy2DL) and [Neural Hydrology](https://doi.org/10.21105/joss.04050).

## Repository Structure
```
.gitattributes
.gitignore
CITATION.cff
environment.yml
LICENSE
README.md
requirements.txt
.ipynb_checkpoints/
aux_functions/
data/
experiments/
results/
```

## Installation

### Using Conda

To create the conda environment, run:
```
conda env create -f environment.yml
conda activate myenv
```

### Using pip
If you are not using Anaconda, you can install the required packages using pip:
```
pip install -r requirements.txt
```

## Usage

### Data Preparation
Ensure that the data files are placed in the `data/` directory. The paths to the data files are specified in the notebooks.

### Running Experiments
Navigate to the `experiments/` directory and run the Jupyter notebooks to train and evaluate the models. For example:
```
jupyter notebook experiments/your_notebook.ipynb
```

### Results
The results of the experiments will be saved in the `results/` directory.

### Configuration
The model hyperparameters and other configurations can be modified in the respective Jupyter notebooks.

## License
This project is licensed under the GNU License - see the LICENSE file for details.

## Citation
If you use this code in your research, please cite:
```
@Article{hess-2024-375,
AUTHOR = {Manoj J, A. and Loritz, R. and Gupta, H. and Zehe, E.},
TITLE = {Can discharge be used to inversely correct precipitation?},
JOURNAL = {Hydrology and Earth System Sciences Discussions},
VOLUME = {2024},
YEAR = {2024},
PAGES = {1--24},
URL = {https://hess.copernicus.org/preprints/hess-2024-375/},
DOI = {10.5194/hess-2024-375}
}
```
