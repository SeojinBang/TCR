<h1 align="center">
    TCR to epitope binding prediction approach using VIBI
</h1>

<br />

## Overview
We illustrate how VIBI can be used to get insights from a model and ensure the safety of a model in a real world application.

## Usage
Download and install the environment from Conda Cloud
```
conda env create SeojinBang/bayberry
conda activate bayberry
```

To learn a black-box model from VDJdb dataset,

```
python original.py --infile data/TCRGP_epitope2cdr3_cleaned_wLabel.txt
```

To make a prediction using the black-box,
```
python original.py --infile data/TCRGP_epitope2cdr3_cleaned_wLabel.txt --indepfile data/netTCR_training_positive_labeled.txt --mode test
```

To learn VIBI to explain the black-box model,
```
python solver.py --infile data/blackboxpred_TCRGP_epitope2cdr3_cleaned_wLabel.txt
```
Run and evaluation on an independent dataset together,
```
python solver.py --infile data/blackboxpred_TCRGP_epitope2cdr3_cleaned_wLabel.txt --indepfile data/netTCR_training_positive_labeled.txt
```
Or from the saved VIBI
```
python solver.py --infile data/blackboxpred_TCRGP_epitope2cdr3_cleaned_wLabel.txt --indepfile data/netTCR_training_positive_labeled.txt --mode test
```

## Data
TCRGP_epitope2cdr3_cleaned_wLabel.txt: VDJdb data preprocessed by TCRGP authors
netTCR_training_positive_labeled.txt: IEDB data preprocessed by netTCR authors

## Credit
[netTCR Repo](https://github.com/mnielLab/netTCR) predicting peptide and TCR interaction.
[TCRGP Repo](https://github.com/emmijokinen/TCRGP) a novel Gaussian process method that can predict if TCRs recognize certain epitopes.

## References
Bang et al. 2019. **Explaining a black-box using Deep Variational Information Bottleneck Approach.** *ArXiv Preprint* [arXiv:1902.06918](https://arxiv.org/abs/1902.06918).

## Contact
Please feel free to contact me by e-mail `seojinb at cs dot cmu dot edu`, if you have any questions.