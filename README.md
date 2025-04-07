# DBDP

**D**eep **B**ackward **D**ynamic **P**rogramming (DBDP) is an open-source Python library currently under active development designed for solving high-dimensional nonlinear partial differential equations (PDEs).

The project implements deep learning methods described in the [research paper](docs/dbdp-research-paper.pdf) by Côme Huré, Huyên Pham, and Xavier Warin, leveraging the classical backward stochastic differential equation (BSDE) representation to efficiently approximate PDE solutions.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/RyanTmi/dbdp
cd dbdp
pip install -r requirements.txt
```

## Structure

[**docs/**](docs/) includes the original research paper.

[**notebooks/**](notebooks/) contains Jupyter notebooks that replicate results from the [research paper](docs/dbdp-research-paper.pdf).

[**models/**](models/) provides pretrained networks and saved model checkpoints.

## Authors

This project is collaboratively developed by :

- Ryan Timeus
- Paul-Antoine Charbit
- Jeremie Vilpellet
