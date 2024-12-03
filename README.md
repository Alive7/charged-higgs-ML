# Signal Background Separation for Charged Higgs Bosons at the LHC

## Train a BDT to separate signal and background events using MC data

This project is designed to take a list of user supplied paths and DSIDs for various signal/background ntuples, process the ntuples into Pandas DataFrames containing relevant physical observables, and train a BDT using the DFs.

### Installation and Use

Simply clone this repo and install its dependancies (scikit-hep, sklearn, and pyroot) to install.

Before running, ensure the desired ntuples paths and event DSIDs are specificied in `main()`. The outputs of this analysis are currently saved under a variaety of sub directories which are created when this script is first run. These can be changed by modifying the appropriate functions; however, care should be taken to ensure that the output of one part of the code is not moved form a location where another part of the code will look for input.

This packge contains two conceptually different familes of functions. Those pertaining to ntuples processing, called with `process_ntuples()`, and those pertaining to BDT training and data exploration which take the processed DFs as input. For this reason, once your desired DSIDs have been processed, the processing functions can be removed from `main()` leaving only the `load_dfs()` functions and subsequent calls. This type of design lends itself to notebooks since the intensive processing step only needs to be run once while the inexpensive AI steps may need to be ran multiple times in the process of hyperparameter tuning or tweaking plots. 

Once `main()` is configured, run from terminal by simply calling with python: `python3 analysis_ssWWjj_run3.py`, or run in a notebook enviornment by creating a notebook in the same directory and importing this modual with `import analysis_ssWWjj_run3`

### More details on editing 'main()'

To configure for a given dataset, the 'main()' function must be edited or reproduced in a notebook enviornment. 'path' specifies the absolute path on lxplus eos to the lists of ntuples for a given run. 'files' identifies the ntuple lists to be considered. DSIDs for the desiered processes must be specified. The can be lumped into a single lists or kept separate if the distiction between singal/brackground or different backgrounds is relevant for your analysis.

Different datasets can be combined by listing the desired DSIDs as long as the associated ntuple lists are specified. For example, a combined MC20 and MC23 analysis may be undertaken by including the appropriate files and paths and listing the DSIDs for a given charged Higgs mass for each run.

## Known Bugs and Further Development

There are several know issues with the current framework. Notably, the baseline selections for the ssWW process are not fully applied and there does not exist a user interface to choose which selections are applied these should be fixed and added. Moreover, the BDT training algorithm itself has not been updated to interface with PD DataFrames or the ROOT based plotting functions. Smaller issues in the data exploration functions exist. For example, the ATLAS plot style is not applied correctly resulting in unfortunate colors on the plots made in ROOT, there is no interface to choose which type of plotting is performed (ratio plots, histograms, hstacks, ect.), and the histogram formatting could be improved overall.

Examination of the souce code reveals two distinct blocks of code which roughly allign with the conceptually distinct parts of this modual. The first block corresponds to the BDT and data exploration part. This code is noticably under commented, makes use of poor variable names, filled with hacks to join code designed for pandas with code designed for numpy, and contains code which does not interface with any other part of the modual. This section requires signifigant development before it should be used in an ATLAS analsis. The second part, alligning with the ntuple processing, is much more complete. It is filled with comments and sanity checks, indicates where potential issues are located, and it uses more expressive variable names. This section harbors known bugs as indicated above, but is much further along and is nearing the state where it can be used in a full analysis.