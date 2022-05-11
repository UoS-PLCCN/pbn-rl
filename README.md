# pbn-rl

Suite of experiments for running Deep Reinforcement Learning for control of Probabilistic Boolean Networks.

# Environment Requirements
- CUDA 11.3+
- Python 3.9+

# Installation
## Local
- Create a python environment using PIP:
    ```sh
    python3 -m venv .env
    source .env/bin/activate
    ```
    For the last line, use `.\env\Scripts\activate` if on Windows.
- Install [PyTorch](https://pytorch.org/get-started/locally/):
    ```sh
    python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    ```
- Install the package and its dependencies dependencies:
    ```sh
    python -m pip install -r requirements.txt
    ```

# Running
- Use `train_ddqn.py` to train a DDQN agent. It's a command line utility so you can check out what you can do with it using `--help`.
    E.g.:
    ```sh
    python train_ddqn.py --time-steps 400_000 --env-name n28 --env envs/n28.pkl
    ```
- Use `train_sb3.py` to train a Stable Baselines 3 agent. It's a command line utility so you can check out what you can do with it using `--help`.
    E.g.:
    ```sh
    python train_sb3.py --time-steps 400_000 --env-name n28 --env envs/n28.pkl
    ```

# Getting Help
Principal developer: [Evangelos Chatzaroulas](mailto:e.chatzaroulas@surrey.ac.uk) ([Alternate e-mail](mailto:evangelos.ch.de@gmail.com)).
