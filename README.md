# In Vino Veritas

(https://github.com/guillaumebrl/In-Vino-Veritas)

Authors : David Admète, Guillaume Bril, Célian Charleau, Hubert de Lesquen,Ruben Didier, Alexandre Gommez. 

This git repository contains all the files needed to participate to the In Vino Veritas ramp data challenge whose goal is to predict the price of a given wine with the best possible accuracy.

## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

### Challenge description

Get started with the [dedicated notebook](in_vino_veritas_starting_kit.ipynb)


### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
