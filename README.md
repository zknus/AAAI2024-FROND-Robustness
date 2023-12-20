

This repository contains the code for our AAAI 2024 accepted paper, *[Coupling Graph Neural Networks with Fractional Order Continuous Dynamics: A Robustness Study]*.

## Table of Contents

- [Requirements](#requirements)
- [Reproducing Results](#reproducing-results)
- [Reference](#reference)
- [Citation](#citation)

## Requirements

To install the required dependencies, refer to the environment.yaml file

## Reproducing Results


For the GMA(Metattack) run the following command:

```bash
python run_metattack_rate_frac.py --dataset cora --function transgrand --block constantfrac --lr 0.005 --dropout 0.4 --input_dropout 0.4 --time 4 --hidden_dim 64 --step_size 1 --runtime 10 --gpu 0 --epochs 800 --patience 100 --batch_norm --method predictor --weight_decay 0.01 --alpha_ode 0.6

python run_metattack_rate_frac_all.py --dataset cora --function transformer --block constantfrac --lr 0.005 --dropout 0.4 --input_dropout 0.4 --time 4 --hidden_dim 64 --step_size 1 --runtime 10 --gpu 0 --epochs 800 --patience 100 --batch_norm --method predictor --alpha_ode 0.6

python run_metattack_rate_frac.py --dataset cora --function belgrand --block constantfrac --lr 0.005 --dropout 0.4 --input_dropout 0.4 --hidden_dim 64 --step_size 0.2 --time 5 --runtime 10 --gpu 1 --epochs 500 --patience 100 --batch_norm --alpha_ode 0.6 --method predictor --no_alpha --weightax 1.0


python run_metattack_rate_frac.py --dataset citeseer --function transformer --block constantfrac --lr 0.005 --dropout 0.4 --input_dropout 0.4 --time 10 --hidden_dim 64 --step_size 1 --runtime 10 --gpu 3 --epochs 800 --patience 100 --batch_norm --method predictor --alpha_ode 0.5

python run_metattack_rate_frac.py --dataset citeseer --function belgrand --block constantfrac --lr 0.005 --dropout 0.4 --input_dropout 0.4 --time 5 --hidden_dim 64 --step_size 1 --runtime 10 --gpu 1 --epochs 800 --patience 100 --batch_norm --method predictor --alpha_ode 0.7 --weight_decay 0.01

python run_metattack_rate_frac.py --dataset citeseer --function transgrand --block constantfrac --lr 0.005 --dropout 0.4 --input_dropout 0.4 --time 10 --hidden_dim 64 --step_size 1 --runtime 10 --gpu 2 --epochs 800 --patience 100 --batch_norm --method predictor --alpha_ode 0.3 --weight_decay 0.01


python run_metattack_rate_frac.py --dataset pubmed --function transgrand --block constantfrac --lr 0.005 --dropout 0.4 --input_dropout 0.4 --time 3 --hidden_dim 64 --step_size 1 --runtime 10 --gpu 0 --epochs 800 --patience 100 --batch_norm --alpha_ode 0.1 --method predictor

python run_metattack_rate_frac.py --dataset pubmed --function belgrand --block constantfrac --lr 0.005 --dropout 0.4 --input_dropout 0.4 --time 3 --hidden_dim 64 --step_size 1 --runtime 10 --gpu 1 --epochs 500 --patience 100 --batch_norm --alpha_ode 0.1 --method predictor

python run_metattack_rate_frac.py --dataset pubmed --function transformer --block constantfrac --lr 0.005 --dropout 0.4 --input_dropout 0.4 --hidden_dim 64 --step_size 1.0 --time 16 --runtime 10 --gpu 0 --epochs 500 --patience 100 --batch_norm --method predictor --alpha_ode 0.1


```


## Reference 

Our code is developed based on the following repo:

The FDE solver is from [torchfde](https://github.com/zknus/torchfde).
The GIA attack method is based on the [GIA-HAO](https://github.com/LFhase/GIA-HAO/tree/master) repo.
The graph neural ODE model is based on the [GraphCON](https://github.com/tk-rusch/GraphCON), [GRAND](https://github.com/twitter-research/graph-neural-pde), and [GraphBel](https://github.com/zknus/Robustness-of-Graph-Neural-Diffusion)   framework.  
The METATTACK and NETTACK methods are based on the [deeprobust](https://github.com/DSE-MSU/DeepRobust) repo.







