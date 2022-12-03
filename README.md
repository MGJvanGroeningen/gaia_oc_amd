# Open Cluster Automatic Membership Determination with Gaia data
This repository presents automatic procedure for determining new members of open clusters. We use a neural network architecture to find additional members of an open cluster based on already established probable members. To train our model, we use astrometric and photometric data of stars obtained with the Gaia space telescope. The repository contains code for downloading and preparing datasets, for training the model and for the visualization of some results. 

## Python environment setup

This code is available as a PyPI package and can be installed with

`pip install gaia_oc_amd`

Alternatively, download the code directly from GitHub with

`git clone https://github.com/MGJvanGroeningen/gaia_oc_amd`

## Data preparation

To use this method, we need to download cluster properties, membership lists and isochrones. To simply get started use:

```python
from gaia_oc_amd import download_startup_data
download_startup_data(save_dir='./data')
```

This downloads cluster parameters provided by [Cantat-Gaudin et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26A...640A...1C/abstract) and membership lists provided by [Cantat-Gaudin et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018A%26A...618A..93C/abstract), [Cantat-Gaudin & Anders (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26A...633A..99C/abstract), [Cantat-Gaudin et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26A...640A...1C/abstract) and [Tarricq et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...659A..59T/abstract). Isochrones are downloaded from the [Padova web interface](http://stev.oapd.inaf.it/cgi-bin/cmd).

In addition, we also want to query the Gaia archive at https://gea.esac.esa.int/archive/ to search for new members, which requires an account for large queries. The code that is used to handle the queries expects a credentials file (with filename 'gaia_credentials' by default), which contains 2 lines for username and password, in the supplied data directory to login to the Gaia archive. This file can be created manually or by running

```python
from gaia_oc_amd import create_gaia_credentials_file
create_gaia_credentials_file(save_path='./gaia_credentials')
```

which prompts for username and password.

## Quick start

With the data and Gaia credentials set up, we can find new members for a cluster with

```python
from gaia_oc_amd import build_sets, evaluate_clusters

cluster_name = 'NGC_1039'

build_sets(cluster_name)
evaluate_clusters(cluster_name)
```

This performs a cone search on the Gaia archive to obtain source data for potential members and uses a pretrained model included in the package to determine their membership status.

## Exploring the code

If you want to explore the methods that are used, be sure to check out the tutorial notebooks in the examples directory. The `quick_tutorial.ipynb` notebook shows a minimal example while the `tutorial.ipynb` notebook walks through the methods step by step.

