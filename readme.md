# Example of Synchronous Observer Design for Velocity-Aided Attitude

This code is an implementation of an example of synchronous observer design for velocity-aided attitude.
It is based on the scientific article available at <https://arxiv.org/abs/2505.19517>

The full bibliography information is provided below.

```latex
@article{van2023synchronous,
  title={Synchronous Models and Fundamental Systems in Observer Design},
  author={van Goor, Pieter and Mahony, Robert},
  journal={arXiv preprint arXiv:2505.19517},
  year={2025}
}
```

The main purpose of the code is to demonstrate the theory developed in the article, but it can be modified for practical applications.

## Requirements

The python libraries used in this code are listed below.
For quick installation of all these packages, use 

```commandline
pip install numpy matplotlib pylieg progressbar2
```

* numpy: `pip install numpy`
* matplotlib: `pip install matplotlib`
* pylie: `pip install pylieg`
* progressbar2: `pip install progressbar2` *Not required.*

## Usage

Simulations can be run using

```commandline
python3 vaa_synchronous.py
```

The code can be configured by commenting or uncommenting various lines.
For any questions or problems, please create a github issue.