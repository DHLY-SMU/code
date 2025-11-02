# Code for "Dynamic involvement of the core gut microbiome XNP_Guild1 in the evolution of gestational diabetes mellitus "

This repository contains the Python analysis scripts used for the computational analyses in the manuscript: "Dynamic involvement of the core gut microbiome XNP_Guild1 in the evolution of gestational diabetes mellitus".

## Overview

The scripts provided are the primary analysis pipelines used to generate the machine learning and network analysis results presented in the manuscript.

## Script Descriptions

The provided Python scripts (`.py`) correspond to the main analyses as follows:

* `Random Forest.py`: Executes the  Random Forest modeling pipeline (Related to Figure 2A).

* `Stable Network Construction.py`: Runs the pipeline to construct the GDM and Non-GDM co-occurrence networks from the OTU tables and identifies the stable core microbial guild (XNP_Guild1) (Related to Extend Figure 5).

* `Topological analysis.py`: Performs topological analysis (e.g., centrality, clustering coefficient, density) on the networks generated in the previous step (Related to Figure 4D and Figure 4E)

* `Z-score Visualization.py`: Compares the topological metrics of the observed networks against 1,000 random networks to generate Z-scores for robustness testing (Related to Figure 3B and Figure 4D).

* `OTU-46 social network changes.py`: Executes the ego-network analysis to compare the local connectivity and neighbor rewiring of the hub microbe `OTU-46` between the GDM and nonGDM states (Related to Extend Figure 7).

## Core Dependencies

The analysis pipelines rely on the following key Python libraries:

* `scikit-learn`
* `shap`
* `pandas`
* `numpy`
* `networkx`
* `matplotlib`
* `seaborn`

## How to Run

1.  Install the required dependencies
2.  Place the necessary data files in the same directory.
3.  Execute the desired analysis script, for example:
    ```sh
    python "Random Forest.py"
    ```

## Data Availability
The newly established prospective cohort data is available from the corresponding author upon reasonable request.
