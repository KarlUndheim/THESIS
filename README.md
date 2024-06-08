# THESIS
Code used for master's thesis

## Acknowledgements
First I want to thank Gefei Kong @Alessiacosmos for her help during the thesis. I also want to credit her work with the simpleview model, which was implemented by her.
Furthermore, ChatGPT has been used as a copilot throughout the coding, as stated in the thesis. Some key implementations not stated in the ai declarations are highlighted in the code.

## Overview
This repository shows the code used for the experiments in my thesis. The code is presented as-is for transparency and documentation purposes.
It is not intended to be executable, as the work was nonlinear and the dataset is not public. 

## Code Documentation
The code is largely uncommented due to its experimental nature. Instead this README provides brief descriptions of the individual scripts or folders to help understand their purpose and functionality without the need to delve into the code itself.

## Repository structure
### preprocess
Contains the code for feature extraction. Except for the BIN features in "mdl_featcalc.py" and the VOX features in "voxel_features.py" the features are just implementations of the benchmark features stated in the thesis.
- mdl_featCalc.py: extraction of radiometric features
- convexhullfeatures.py: extraction of CA, CV, CD, Hrange, Hmax, Hmean, Hstd
- penetration_density.py: extraction of penetration and density features D_i
- voxel_features.py: extraction of the voxel-features
- std_mean.py: creates std and mean for the feature tables so that they can be used with neural networks.

### plots
Contains the code used to plot most figures in the thesis.

### ML
Contains the code for the model comparison and feature experiments
- cfg_ml_rf.yaml: config file for choosing features and species grouping for the MLP
- EDA.ipynb: used for exploratory data analysis and plotting histograms, correlations matrices and boxplots.
- MLP: Complete implementation of the MLP
- modelcomparison.ipynb: experiments for XGBoost and RF for the model comparison experiments
- feature_experiments: experiments for the BIN and VOX features using XGBoost

### simpleview
Contains the main files for the simpleview model. Thanks to Gefei Kong for providing the code. Note that though these are implementations of "DetailView" I excluded their modification for my experiments so the model performs like SimpleView
- cfgf_dfconvnextt...: config file for running the experiments. Here all parameters, species, resolution and features are selected.
- dataset_titan_dv: creation of the dataset
- detailview: model implementation. FEATURE BRANCH ADDED HERE
- main_titan_f: main script
- train_titan_f: training script

