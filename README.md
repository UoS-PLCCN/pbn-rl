# DDQN with PER

We study the ability of a Double Deep Q-Network (DDQN) with Prioritized Experience Replay (PER) in learning control strategies within a finite number of steps that drive a probabalistic Boolean network (PBN) towards a target state, typically an attractor. The control method is model-free and does not require knowledge of the network's underlying dynamics, making it suitable for applications where inference of such dynamics is intractable. We have tried the method on a number of networks, synthetic PBNs but also on PBN generated directly from real gene expression profiling data.

This project is based on the work found in our paper at: https://link.springer.com/chapter/10.1007/978-3-030-65351-4_29
 
An extended version of the above paper can be found at: https://arxiv.org/abs/1909.03331 

# Licensing 
See [LICENSE](https://gitlab.com/af00150/ddqn-with-per/-/blob/master/LICENSE) file for licensing information as it pertains to
files in this repository. 

# Environment Requirements
- CUDA 11.2+
- Python 3.9+

# Installation
- Create a python environment using PIP:
    ```sh
    python3 -m pip install virtualenv
    python3 -m virtualenv .env
    source .env/bin/activate
    ```
    For the last line, use `.\env\Scripts\activate` if on Windows.
- Install the package and its dependencies dependencies
    ```sh
    python -m pip install -e .
    ```

# Install Environment within Docker
* TBD

# Documentation
*In development

# Running
- Adjust the profiles under `pbn_control/training/Environments.yml` as well as any Agent settings in `pbn_control/training/Environments`.
- Run `python pbn_control/training/runner.py`. Adjust the training and testing flags in lines 340 and 341 respectively to enable model training or model testing during the run.

# Getting Help
Principal developer: [Evangelos Chatzaroulas](mailto:ec00727@surrey.ac.uk) ([Alternate e-mail](mailto:evangelos.ch.de@gmail.com)).

# Associated Publications
Deep Reinforcement Learning for Control of Probabilistic Boolean Networks

- https://link.springer.com/chapter/10.1007/978-3-030-65351-4_29

Deep Reinforcement Learning for Control of Probabilistic Boolean Networks (Extended)

- https://arxiv.org/abs/1909.03331
