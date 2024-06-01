TODO: Write actual README

# worms - Worm Simulation and Analysis Library

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description

This library provides tools for simulating worm movement, training predictors, and analyzing the results. It includes support for YOLO-based prediction and various simulation controllers.

## Features

- Worm movement simulation
- YOLO-based prediction
- Logging and analysis tools
- CSV, logging, and YOLO controllers

## Installation

Clone the repository and install the necessary Python dependencies. This project requires Python 3.12 or higher.

Step 1 - Install mamba:
    Install 'Miniforge' from this [link](https://github.com/conda-forge/miniforge), make sure to download the right version (that match the OS and CPU of the computer).
    If asked during installation: add to PATH.

Step 2 - verify that mamba is installed correctly::
    1. open terminal/command prompt
    2. enter 'mamba -h'
    
    if no error is encoutered than mamba is installed correctly.

Step 3 - create the project environment:
    1. copy the path of the 'installation.py' file (found in the project directory)
    2. open terminal/command prompt
    3. enter 'mamba activate base'
    4. enter 'python PATH' where PATH is the path from step 1 
     

## Usage

Use the provided Jupyter notebooks for various tasks:

- [initialize_experiment.ipynb](create_experiment.ipynb) to set up a new experiment
- [create_yolo_dataset.ipynb](create_yolo_dataset.ipynb) to create a dataset for YOLO
- [predictor_training.ipynb](predictor_training.ipynb) to train the predictor
- [simulate.ipynb](simulate.ipynb) to run the simulation
- [yolo_training.ipynb](yolo_training.ipynb) to train YOLO

## Contributing

Contributions are welcome. Please submit a pull request with your changes and include a detailed description of your improvements.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

Please open an issue in the GitHub repository if you have any questions or feedback.