<h1 align="center">
    [Temp title] TCR analysis
</h1>

<br />

## Overview
This is a deep learning model for predicting TCR-epitope binding

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
netTCR repo
TCRGP repo

## References
Our recent paper on arXiv or bioArXiv??!

## Contact
Please feel free to contact me by e-mail `seojinb at cs dot cmu dot edu`, if you have any questions.