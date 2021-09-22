# Automated Single Cell Tuner (ASCT)
The aim of this tool is the automate the parameter selection process for biologically realistic neuronal models using [SBI](https://www.mackelab.org/sbi/).

## Installation:
There are two options for using ASCT; You can run your single cell optimization problem online at [cyneuro](https://engineering.missouri.edu/research/research-initiatives/cyneuro/) or you can download the source-code and run the project directly.

## Running paramter inference.
### The CLI interface.
ASCT is a CLI based tool which takes a configuration file and command line arguments as inputs. For basic usage, run the command ```python optimize_cell.py -h```. Which will prodce the following output:
```
usage: optimize_cell.py [-h] [-g] [-c] [-l] [-n N] config_dir [save_dir]

Uses SBI to find optimal parameter sets for biologically realistic neuron
simulations.

positional arguments:
  config_dir  the optimization config file directory
  save_dir    [optional] the directory to save figures to

optional arguments:
  -h, --help  show this help message and exit
  -g          displays graphics
  -c          compiles modfiles
  -l          store log files
  -n N        the number of found parameters to show (must be less than the
              threshold sample size in the optimization config file)
```
From here the user has several options which they can specify depending on their individual needs.

### Configuration file setup.
The only required argument for ASCT is a user defined json configuration file. This is a file which specifies all relevant information which is needed to infer the parameters a user is interested in. There are five subsections required in every configuration file for ASCT: ```manifest```, ```optimzation_settings```,```run```,```conditions```, and ```optimzation_parameters```. Each subsection has a variety of options which the user can set for a general run, some of which are required and some which are not. Here is the general breakdown of a config file by section.
#### ```manifest```
* Required Parameters:
    1. ```"job_type"```: Specifies if a user wants to optimize parameters from provided current response information or if they would like to validate against a ground truth model. Can be set to either ```"ground_truth"``` or ```"from_data"``` respectively.
    2. ```"architecture"```: Specifies what optimization architecture the user wishes to use with this model. There are three options: ```"convolution"```,```"hybrid"```, and ```"summary"```. ```"convolution"``` uses a Convolutional Neural Network (CNN) to learn a user defined number of features to encode voltage response data, ```"summary"``` uses a user defined function for the same purpose, and ```"hybrid"``` uses a combination of both.
