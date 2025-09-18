# About  
*cubbyhouse* is a Python package for the design and optimisation of timber-framed structures using a stock of reclaimed timber elements. 

# Installation: Python  
To install *cubbyhouse* in a local development environment, follow these steps:  

```bash
# Create a virtual environment
python -m venv env  

# Activate the virtual environment
# On Windows:
env\Scripts\activate.bat
# On macOS/Linux:
source env/bin/activate  

# Install dependencies
pip install -r requirements.txt  

# Install in editable mode
pip install --editable .
```

# Getting Started: Python
Refer to the demo scripts in the [examples_py](examples_py) folder for package operation. These provide sample workflows for defining a reclaimed timber stock, generating structural frames, and running optimisation routines.

# Getting Started: Rhino 8
Demo scripts for a *cubbyhouse* interactive user interface are provided in the [examples_gh](examples_gh) folder. These require Rhino 8 and for the *cubbyhouse* package and dependencies to be installed in the Rhino Python 3 runtime environment. 
- To install the *cubbyhouse* package: Copy the src/cubbyhouse folder into "C:\Users\<username>\.rhinocode\py39-rh8\Lib\site-packages"
- To install dependencies: Follow instructions [here](https://developer.rhino3d.com/guides/rhinopython/python-packages/) or navigate to the Rhino 8 Python installation and install from the terminal "C:\Users\<username>\.rhinocode\py39-rh8"


# License
cubbyhouse is an open source engineering tool provided under an MIT License. Please refer to the [LICENSE](LICENSE) file for more information.

Please note that this software has been developed by student and academic engineering researchers, *not* by chartered or registered engineers.

# Acknowledgements
This package has been developed from research projects supported by the University of Queensland.

If you use cubbyhouse for projects or scientific publications, please consider citing our journal paper:

> Gattas, J.M., Ottenhaus, L.-M., Liu, H., & Xie, Y.M. (2025). Design and optimisation of timber-framed structures using a stock of reclaimed elements. *Automation in Construction*. [doi:10.1016/j.autcon.2025.106527](https://doi.org/10.1016/j.autcon.2025.106527).

Results reported in this paper are available in the [results](results) folder.