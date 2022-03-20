# MSc Informatics Dissertation - University of Edinburgh

This repo contains the thesis and code for my MSc dissertation, done through the School of
Informatics at the University of Edinburgh. My supervisor was Dr. Michael Gutmann.

# Thesis

The title of my thesis is **Fast and Scalable Factor Analysis Algorithms for Bayesain Deep 
Learning**. 

It can be found [here](https://github.com/ayeright/swafa/blob/master/thesis/main.pdf).

# Algorithms

## Online SGA FA

Algorithm 1 from the thesis is implemented in the 
`OnlineGradientFactorAnalysis` class
[here](https://github.com/ayeright/swafa/blob/master/swafa/fa.py).

## Online EM FA

Algorithm 2 from the thesis is implemented in the 
`OnlineEMFactorAnalysis` class
[here](https://github.com/ayeright/swafa/blob/master/swafa/fa.py).

## VIFA

Algorithm 3 from the thesis is implemented in the 
`FactorAnalysisVariationalInferenceCallback` class
[here](https://github.com/ayeright/swafa/blob/master/swafa/callbacks.py).

# Experiments

The main experiments from the thesis are defined in the `dvc.yaml` file.

After installing the dependencies in `requirements.txt`, the experiments can be reproduced by
running the following command from root of the project:
```bash
dvc repro
```

In order to reproduce a specific experiment, run the following command:
```bash
dvc repro <stage name from dvc.yaml>
```

The stages correspond to the experiments in the following section of the thesis:

* `online_fa` an `online_fa_analysis`: Section 4.4.1
* `linear_regression_vi`: Section 5.2.2
* `neural_net_predictions`: Section 5.2.3

If you want to change any of the parameters for the experiments, you can do so in `params.yaml`.