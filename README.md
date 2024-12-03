# Startup Instructions

Run from terminal by simply calling with python: python3 analysis_run2+3.py

To configure for a given dataset, the main() function must be edited. path specifies the absolute path on lxplus eos to the lists of ntuples for a given run. files identifies the ntuple lists to be considered. DSIDs for the desiered processes must be specified. The can be lumped into a single lists or kept separate if the distiction between singal/brackground or different backgrounds is relevant for your analysis.

Different datasets can be combined by listing the desired DSIDs as long as the associated ntuple lists are specified. For example, a combined MC20 and MC23 analysis may be undertaken by including the appropriate files and paths and listing the DSIDs for a given charged Higgs mass for each run.

