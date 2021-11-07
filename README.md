# Machine Learning Assignment, AIMS CDT 2021

*Author: Shreshth Malik*

## Overview

This repository implements a neural network from scratch using linear algebra routines and autodiff using PyTorch.

## Data

We use the [Iris Data Set](https://www.kaggle.com/arshid/iris-flower-dataset) in this work. This consists of 150 records of flowers, each with 4 numerical features (sepal and petal lengths and widths), and a corresponding class label. This is a multi-class classification problem (3 classes to distinguish). We randomly hold out 20% of the data to use as test data for model evaluation.

## Usage

First install the requirements: `pip install -r requirements.txt`. 

Optional arguments (to vary architecture and optimisation hyperparameters) are set using command line arguments, with `main.py` as the entry point. Run `python main.py -h` to check what hyperparameters are customisable. 

To reproduce the results in the report (training curves and final test metrics), simply run the program (from the same directory) using the default settings:
```
python3 main.py
```
