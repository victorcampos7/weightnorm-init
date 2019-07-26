# How to Initialize your Network? Robust Initialization for WeightNorm & ResNets

This repository contains code for [How to Initialize your Network? Robust Initialization for WeightNorm & ResNets](https://arxiv.org/abs/1906.02341).

## Abstract

Residual networks (ResNet) and weight normalization play an important role in various deep learning applications. However, parameter initialization strategies have not been studied previously for weight normalized networks and, in practice, initialization methods designed for un-normalized networks are used as a proxy. Similarly, initialization for ResNets have also been studied for un-normalized networks and often under simplified settings ignoring the shortcut connection. To address these issues, we propose a novel parameter initialization strategy that avoids explosion/vanishment of information across layers for weight normalized networks with and without residual connections. The proposed strategy is based on a theoretical analysis using mean field approximation. We run over 2,500 experiments and evaluate our proposal on image datasets showing that the proposed initialization outperforms existing initialization methods in terms of generalization performance, robustness to hyper-parameter values and variance between seeds, especially when networks get deeper in which case existing methods fail to even start training. Finally, we show that using our initialization in conjunction with learning rate warmup is able to reduce the gap between the performance of weight normalized and batch normalized networks.


## Dependencies

This project was developed with Python 3.6. Dependencies can be installed using the provided `requirements.txt` file.


## Running the code

Figure 1 in the paper can be reproduced with `src/notebooks/synthetic_data_experiment.ipynb`. The rest of experiments can be reproduced with `src/train.py`.

For instance, the following command would train an MLP on MNIST using WeightNorm with the proposed initialization:

`cd src && python -m train --dataset mnist --nn mlp --weight_norm --init proposed_orthogonal`


## Citation

If you use this in your work, please cite

```
@article{arpit2019initialize,
  title={How to Initialize your Network? Robust Initialization for WeightNorm \& ResNets},
  author={Arpit, Devansh and Campos, Victor and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1906.02341},
  year={2019}
}
```
