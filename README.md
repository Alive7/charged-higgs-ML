# Signal Background Separation for Charged Higgs Bosons at the LHC

## Train a BDT to separate signal and background events using MC data

This project is designed to take a list of user supplied paths and DSIDs for various signal/background ntuples, process the ntuples into Pandas DataFrames containing relevant physical observables, and train a BDT using the DFs.

### Installation and Use

Simply clone this repo and install its dependencies (scikit-hep, sklearn, and pyroot) to install.

Before running, ensure the desired ntuples paths and event DSIDs are specified in `main()`. The outputs of this analysis are currently saved under a variety of sub directories which are created when this script is first run. These can be changed by modifying the appropriate functions; however, care should be taken to ensure that the output of one part of the code is not moved from a location where another part of the code will look for input.

This package contains two conceptually different families of functions. Those pertaining to ntuples processing, called with `process_ntuples()`, and those pertaining to BDT training and data exploration which take the processed DFs as input. For this reason, once your desired DSIDs have been processed, the processing functions can be removed from `main()` leaving only the `load_dfs()` functions and subsequent calls. This type of design lends itself to notebooks since the intensive processing step only needs to be run once while the inexpensive AI steps may need to be ran multiple times in the process of hyperparameter tuning or tweaking plots. 

Once `main()` is configured, run from terminal by simply calling with python: `python3 analysis_ssWWjj_run3.py`, or run in a notebook environment by creating a notebook in the same directory and importing this module with `import analysis_ssWWjj_run3`

### More details on editing `main()`

To configure for a given dataset, the `main()` function must be edited or reproduced in a notebook environment. `path` specifies the absolute path on lxplus eos to the lists of ntuples for a given run. `files` identifies the ntuple lists to be considered. DSIDs for the desired processes must be specified. They can be lumped into a single lists or kept separate if the distinction between signal/background or different backgrounds is relevant for your analysis.

Different datasets can be combined by listing the desired DSIDs as long as the associated ntuple lists are specified. For example, a combined MC20 and MC23 analysis may be undertaken by including the appropriate files and paths and listing the DSIDs for a given charged Higgs mass for each run.

## Known Bugs and Further Development

There are several known issues with the current framework. Notably, the baseline selections for the ssWW process are not fully applied and there does not exist a user interface to choose which selections are applied these should be fixed and added. Moreover, the BDT training algorithm itself has not been updated to interface with PD DataFrames or the ROOT based plotting functions. Smaller issues in the data exploration functions exist. For example, the ATLAS plot style is not applied correctly resulting in unfortunate colors on the plots made in ROOT, there is no interface to choose which type of plotting is performed (ratio plots, histograms, hstacks, ect.), and the histogram formatting could be improved overall.

Examination of the source code reveals two distinct blocks of code which roughly align with the conceptually distinct parts of this module. The first block corresponds to the BDT and data exploration part. This code is noticeably under commented, makes use of poor variable names, filled with hacks to join code designed for pandas with code designed for numpy, and contains code which does not interface with any other part of the module. This section requires significant development before it should be used in an ATLAS analysis. The second part, aligning with the ntuple processing, is much more complete. It is filled with comments and sanity checks, indicates where potential issues are located, and it uses more expressive variable names. This section harbors known bugs as indicated above, but is much further along and is nearing the state where it can be used in a full analysis.