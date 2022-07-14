# Open Cluster Automatic Membership Determination with Gaia data
This repository is part of a Master's research project (2021-2022) in which an automatic procedure for determining members of open clusters was developed. We use a neural network architecture to find additional members of an open cluster based on already established probable members. To train our model, we use astrometric and photometric data of stars obtained with the Gaia space telescope. The repository contains code for downloading and preparing datasets, for training the model and for the visualization of some results. 

## Python environment setup

If conda is installed, an environment for running the code can be created by running 

`conda create -n gaia_oc_amd python=3.8`

`conda activate gaia_oc_amd`

`pip install -r requirements.txt`

## Data preparation

### Member catalogues

As the method requires knowledge of already established open cluster members, we first need to create a dataset of known members. Two extensive membership catalogues can be downloaded with

`python download_members_and_cluster_params.py`,

which will download membership lists for 1229 open clusters obtained by [Cantat-Gaudin et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018A%26A...618A..93C/abstract) (CG18) and membership lists for 389 open clusters obtained by [Tarricq et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...659A..59T/abstract) (T22). Throughout the project, the CG18 members have been used for training the model and the T22 members were used to compare the obtainted additional members against. In general, the code requires (i) the source identity, (ii) the cluster to which they belong and (iii) the membership probability for the members.

### Cluster parameters

As the filename suggest, `download_members_and_cluster_params.py` also downloads a list of parameters for 2017 known clusters, obtained by [Cantat-Gaudin et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26A...640A...1C/abstract). We use the parameters to calibrate cone searches on clusters, which obtain data for stars in the vicinity of the cluster. These stars are selected such that they contain both candidate members, on which the trained model is applied to verify their membership status, and non-members, which serve as negative examples during training.

### Gaia archive credentials

We download this stellar data from the Gaia archive https://gea.esac.esa.int/archive/, which requires an account for large queries. The code that is used to handle the queries expects a credentials file (with filename 'gaia_credentials' by default), which contains 2 lines for username and password, in the supplied data directory to login to the Gaia archive. This file can be created by running

`python create_gaia_credentials_file.py [username] [password]`.

### Isochrones

The method also employs the theoretical isochrone of the cluster, which defines the colour and magnitude of stars with the same age and initial chemical composition for different initial masses. Isochrone data can be downloaded from http://stev.oapd.inaf.it/cgi-bin/cmd. For the photometric system, use the option 'Gaia EDR3 (all Vegamags, Gaia passbands from ESA/Gaia website)'. By default, the code expects a 'isochrones.dat' file, containing the data for isochrones with various ages, in the data directory. 

Instead of manually downloading the isochrones, use

`python download_isochrones.py`

to download data for 400 (the maximum for a single query) isochrones between a log age of 6 and 9.99 and a default metal fraction Z = 0.0152.

## Creating member, candidate and non-member sets

With all of this set up (i.e. member data, cluster parameters, gaia archive credentials and isochrone data), we can prepare the various sets of sources (members, candidates, non-members) which will be used to create the datasets on which the model will be trained. To do this for one or more clusters, run

`python build_sets.py [clusters]`,

where the clusters argument is either the name of a single cluster as given in the cluster parameters file, e.g. NGC_2509, or a file that contains a cluster name on every line. This script performs a cone search on each cluster provided and divides the sources in each cone dataset in members, candidates and non-members. Various optional arguments are available for adjusting the cone search parameters as well as the candidate selection conditions.

## Training the model

**Note: parameters of an already trained model are available in the 'deep_sets_model' folder. You can skip this step if you only want to use the trained model.**

Training the model can be done with

`python train_deep_sets_model.py [clusters]`,

where the clusters argument is again either the name of a single cluster or a file that contains the names of multiple clusters. The model can thus be trained on a single cluster or multiple clusters, but training on multiple clusters will result in a more general model. Optional arguments for `train_deep_sets_model.py` include both hyperparameters and parameters that determine the training/validation datasets, such as how many members and non-members to use per cluster.

## Candidate evaluation

After training, we can use the model to find additional members of open clusters. To evaluate the candidate members of one or more clusters, use

`python evaluate_clusters.py [clusters]`,

where the clusters arguments works the same as before. By default, this function looks for model parameters in a 'deep_sets_model' directory. Candidate membership probabilities are saved in a candidates.csv file in the data/results directory of the corresponding cluster. The script also creates a number of plots that show the distribution of the obtained members and compares it to either the training members or a supplied set of comparison members.

## Exploring the code

In case you want to explore the main functions that are used for this method, the `tutorial.ipynb` notebook walks through the method step by step.

