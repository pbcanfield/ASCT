# Automated Single Cell Tuner (ASCT)
The aim of this tool is the automate the parameter selection process for biologically realistic neuronal models using [SBI](https://www.mackelab.org/sbi/).

## Installation:
There are two options for using ASCT; You can run your single cell optimization problem online at [cyneuro](https://engineering.missouri.edu/research/research-initiatives/cyneuro/) or you can download the source-code and run the project directly.

## Running parameter inference.
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
The only required argument for ASCT is a user defined json configuration file. This is a file which specifies all relevant information which is needed to infer the parameters a user is interested in. There are five subsections required in every configuration file for ASCT: ```manifest```, ```optimzation_settings```,```run```,```conditions```, and ```optimzation_parameters```. Each subsection has a variety of options which the user can set for a general run, some of which are required and some are conditional based on the user's job. Here is the general breakdown of a config file by section.
#### ```"manifest"```
* Required Settings:
    * ```"job_type"```: Specifies if a user wants to optimize parameters from provided current response information or if they would like to validate against a ground truth model. Can be set to either ```"ground_truth"``` or ```"from_data"``` respectively.
    * ```"architecture"```: Specifies what optimization architecture the user wishes to use with this model. There are three options: ```"convolution"```,```"hybrid"```, and ```"summary"```. ```"convolution"``` uses a Convolutional Neural Network (CNN) to learn a user defined number of features to encode voltage response data, ```"summary"``` uses a user defined function for the same purpose, and ```"hybrid"``` uses a combination of both.
    * ```"template_name"```: Specifies the name of the [NEURON](https://neuron.yale.edu/neuron/) HOC template to use for optimization. 
    * ```"template_dir"```: The directory where the above template HOC file is stored on the local machine.
    * ```"modfiles_dir"```: The directory where the mod files for the above neurn template are stored on the local machine.
* Conditional Settings: 
    * If ```"job_type"``` is set to ```"ground_truth"```:
        1. ```"target_template_name"```: Specifies the name of the target [NEURON](https://neuron.yale.edu/neuron/) HOC template to use for optimization.
        2. ```"target_template_dir"```: The directory where the above target template HOC file is stored on the local machine.
    * If ```"job_type"``` is set to ```"from_data"```:
        1. ```"input_data"```: The CSV file storing the current injection responses for the cell you wish to tune. The file should be comma-seperated where each column corresponds to a seperate current injection value. There should be **exactly 1024 data points** in each column and each data point should be a voltage (mV) corresponding to the membrane potential at each time step of the experiment.
#### ```"summary"``` (Only required if ```"architecture"``` is set to ```"summary"``` in ```"manifest"```)
* Required Settings:
    * ```"summary_file"```: This is the file which specifies user defined summary statistics functions. Each function must be implmented by the user and must two [numpy](https://numpy.org/) arrays of length 1024 for the first two positional arguments. The first array stores the membrane voltage of a given cell and the second is the time vector.
    * ```"function_name"```: This is the name of the function to use in the provided file.
* Conditional Settings: All other parameters in this section are conditional based on the function definition. They will be passed into the function as kwargs.
#### ```"Run"``` (all time units are in ms)
* Required Settings:
    * ```"tstop"```: The duration of each simulation.
    * ```"delay"```: The time before current injection is provided.
    * ```"duration"```: The length of the current injection

#### ```"Conditions"``` (all voltage measurements are in mV)
* Required Settings:
    * ```"v_init"```: The initial voltage of the simulation.
#### ```"optimization_settings"``` 
* Required Settings:
    * ```"num_simulations"```: The number of samples to take within the prior dirstribution.
    * ```"num_rounds"```: The number of rounds if using [multi-round inference](https://www.mackelab.org/sbi/tutorial/03_multiround_inference/).
    * ```"workers"```: The number of workers to use for inference.
* Conditional Settings:
    * If ```"architecture"``` is set to ```"convolution"``` or ```"hybrid"```:
        * ```"features"```: The number of features the CNN will learn.
#### ```"optimization_settings"``` 
* Required Settings:
    * ```"num_simulations"```: The number of samples to take within the prior dirstribution.
    * ```"num_rounds"```: The number of rounds if using [multi-round inference](https://www.mackelab.org/sbi/tutorial/03_multiround_inference/).
    * ```"workers"```: The number of workers to use for inference.
* Conditional Settings:
    * If ```"architecture"``` is set to ```"convolution"``` or ```"hybrid"```:
        * ```"features"```: The number of features the CNN will learn.
#### ```"optimization_paraemeters"``` 
* Required Settings:
    * ```"current_injections"```: A list of current injection values to optimize at (in nA).
    * ```"parameters"```: The list of parameters to optimize, this should match the exact name of the respective HOC mechanism.
    * ```"lows"```: A list of the lower bound for each of the above parameters.
    * ```"highs"```: A list of the upper bounds for each of the above parameters.
* Conditional Settings:
    * If ```"job_type"``` is set to ```"ground_truth"```:
        * ```"ground_truth"```: A list of the model ground truth parameters.
