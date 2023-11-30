# brain_cell_segmentation_3d_images
This repository currently contains the code for my master thesis.

## Folders

### training_resources
This folder contains:
- The training data for the neural network model

### scripts  
In this folder test scripts that execute the code from our trpseg package can be placed.
This folder currently contains a script to segment the ventricle and a script to compute evaluation metrics for segmentation results

Here you could also place your own scripts.

If you want to execute scripts from this folder via commandline then you should do the following:

Assume you want to execute the script test_script.py, which is placed inside the scripts folder.

1. Change your current working directory to the project folder (the folder containing the scripts folder)
2. Use the command `python -m scripts.test_script`

### trpseg
This is our main python package that contains all the important code.
This folder contains:
- The GUI Code written with PyQt6
- The code for the segmentation and quantification of:
  - Pars Tuberalis Cells
  - Blood Vessel Wall
  - Filled Blood Vessels (blood vessel lumen)
- Utility code used by the segmentation and quantification algorithms (includes vasculature graph generation)
- Resource folder with machine learning models (trained neural network models / trained random forest models)


## Installation

1. Install python 3.11.x or 3.10.x (Also older python versions should work, but were not tested with this project.)
2. Clone or Download this repository
3. Install the required python packages. See sections: [Install packages using a virtual environment](#install-packages-using-a-virtual-environment) ,
   [Install packages without virtual environment](#install-packages-without-virtual-environment)

   We recommend to install the packages in a python virtual environment.
   So there will be no problems if older or newer versions of the required packages are already installed globally on your system
4. Execute the code. See section [How to execute?](#how-to-execute)

### Install packages using a virtual environment
More Information: https://docs.python.org/3/library/venv.html

On Windows do the following in the commandline:

- Create virtual environment in a folder of your choice: `python -m venv c:\path\to\myenv`
- Activate the environment `<path_to_venv_folder>\Scripts\activate.bat`
- Install required packages (change current working directory to this project folder): `pip install -r requirements.txt`


On Unix-based systems do the following in the bash:

- Create virtual environment in folder of your choice: `python -m venv /path/to/myenv`
- Activate the environment `source <path_to_venv_folder>/bin/activate`
- Install required packages (change current working directory to this project folder): `pip install -r requirements.txt`

After that you are ready to execute the python code. See section [How to execute?](#how-to-execute).
When you are finished you can call `deactivate` to leave the virtual environment.


### Install packages without virtual environment

Use the commandline (Windows) or bash (Unix).
- change the current working directory to this project folder (the folder containing the requirements.txt)
- call `pip install -r requirements.txt`

After that you are ready to execute the python code. See section [How to execute?](#how-to-execute).


## How to execute?

### Using the commandline or bash
If you created a virtual environment. First activate it using
`<path_to_venv_folder>\Scripts\activate.bat` on Windows commandline or
`source <path_to_venv_folder>/bin/activate` on Unix bash.
When you are finished executing the python code you can leave the virtual environment again by calling `deactivate`.

To start the GUI, execute the file gui_main.py: `python gui_main.py`
You can add and execute your own scripts here too: `python my_script.py`

To execute scripts inside the scripts folder see section [scripts](#scripts) above.


### Using PyCharm or other IDEs

In PyCharm just click `Open...` and select the project folder. If you created a virtual environment yourself then you can
place the virtual environment folder into your project folder. Name of this folder should be venv. Else you can let PyCharm create a
virtual environment. After that you should be ready to execute or debug the Code using PyCharm.

Also, other IDEs (e.g. Visual Studio Code) could be used. However, we only tested our project in PyCharm and do not give more details on how to use
our project in other IDEs.