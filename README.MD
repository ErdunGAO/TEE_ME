# A Variational Framework for Estimating Continuous Treatment Effects with Measurement Error

This repository contains an implementation of the average dose-reponse function estimation methods with measurement error described in ["A Variational Framework for Estimating Continuous Treatment Effects with Measurement Error
"](https://openreview.net/forum?id=S46Knicu56).

If you find it useful, please consider citing:
```bibtex
@inproceedings{
gao2024a,
title={A Variational Framework for Estimating Continuous Treatment Effects with Measurement Error},
author={Erdun Gao and Howard Bondell and Wei Huang and Mingming Gong},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=S46Knicu56}
}
```

## Examples
In this repository, please start by generating all the data required for the experiments by running `bash scripts/simu_data_gen.sh`. Once the data generation is complete, you can proceed by running the main script with `python main.py`.

## Requirements
Use `conda env create -f environment.yml` to create a `torch` conda environment.