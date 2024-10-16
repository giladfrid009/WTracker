# WTracker

## Description

This library provides tools for worm detection and movement prediction, training predictors, and analyzing the results. It includes support for YOLO-based prediction and various simulation controllers.

## Features

- Real-time Worm detection and movement prediction
- Logging and analysis tools
- CSV, logging, and YOLO controllers

## Documentation

* There is an [Official Documentation](https://giladfrid009.github.io/WTracker/) website availabe of the entire API.
* The library is fully documented within the code base.
* Workflow files have elaborate documentation for usage.

## Installation

### Download the Repository
Download the project [repository](https://github.com/giladfrid009/Bio-Proj) (by clicking on code -> download zip) and extract the files in the desired location.

### Environment Installation
**Step 1**- Install mamba:
Install 'Miniforge' from this [link](https://github.com/conda-forge/miniforge), make sure to download the right version (that match the OS and CPU of the computer).  
    If asked during installation: add to PATH.  
    \* if unsure, use this [link](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe) to download mamba.

**Step 2** - verify that mamba is installed correctly:  
1. Navigate to the folder into which the library was download.    
2. Open terminal/command prompt.  
3. Enter - 'mamba -h'.  
    if no error is encountered then mamba is installed correctly.

**Step 3** - create a new environment:  
1. Enter the following command - "mamba create -n bio-proj python=3.12".    
2. Enter the command - 'mamba init'.  
	\* You can choose another name (not only 'bio-proj'). If you do , you will need to change the name field in the 'requirements.yaml' file as well.  

**Step 4** - Activate the environment:  
1. Enter the command - 'mamba activate bio-proj'.  
	* If you used another name for the environment, replace 'bio-proj' with the name you have chosen.   

**Step 5** - Installing Pytorch:  
1. Head to the pytorch website [here](https://pytorch.org/get-started/locally/), there you will find a table to select your configuration, select the following:    
	1. PyTorch Build = stable   
	2. OS - the operating system of the computer \[Windows/Linux\]  
	3. Package - the package manager \[conda\]  
	4. Language - the programming language \[Python\]  
	5. Compute Platform - if the computer has GPU select the newest version of CUDA \[12.1 was tested\], otherwise select CPU.    
2. Copy the command below the table and enter it in the terminal/command prompt   
3. Wait till the installation is complate. That might take a while.    

**Step 6** - Install the rest of the libraries:  
  Enter the command - 'mamba env update -f requirements.yaml -n bio-proj'   


### Install the Development Environment

To run the project we recommend 'Visual Studio Code' (also referred as VS Code), a free IDE. Basic usage videos and documentation can be found [here](https://code.visualstudio.com/docs/getstarted/introvideos).

You can download and install VS Code from [here](https://code.visualstudio.com/download).

To set up VS Code for the project you need to install several extensions. 
Follow this [link](https://code.visualstudio.com/docs/editor/extension-marketplace) to learn how to install extensions. 
The extensions needed are:
- [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)

\* Some extensions may be already installed by default.


## Usage

Refer to the variouse '\.ipynb' files for usage for each workflow.

## License

The code is licensed under the **GPL v3.0 License**.

**_TL;DR_** Here's what the license entails:

```markdown
1. Anyone can copy, modify and distribute this software.
2. You have to include the license and copyright notice with each and every distribution.
3. You can use this software privately.
4. You can use this software for commercial purposes.
5. If you dare build your business solely from this code, you risk open-sourcing the whole code base.
6. If you modify it, you have to indicate changes made to the code.
7. Any modifications of this code base MUST be distributed with the same license, GPLv3.
8. This software is provided without warranty.
9. The software author or license can not be held liable for any damages inflicted by the software.
```

For more details see the [license](LICENSE) file.

## Contact

Please open an issue in the GitHub repository if you have any questions or feedback.
