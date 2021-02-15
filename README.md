# BAROCCO
This code accompanies the paper "Balancing Rational and Other-Regarding Preferences in Cooperative-Competitive Environments".

The code is divided into 4 branches, corresponding to the combinations of two envirionments (Harvest and Eldorado) and two frameworks (Q-learning and Actor-Crirtc). The main branch is empty.

Forge.py is the main file, run it to train algorithms. Change algorithms, settings, and hyperparameters in configs.py. Make figures from logs by running figures.py (prettier in Actor-Critic branches).

The folder 'forge' contains all the meat, as follows. 'engine' contains all neural networks. 'ethyr' contains all optimization stuff required to train the networks. 'blade' contains all environment logic. 'trinity' contains high-level classes: pantheon.py and pantheon_lm.py contain high-level logic to update networks, sword.py contains high-level logic for agent-environment interaction to collect rollouts, smith.py alternates between interaction with environment and network updates -- either sequentially (Actor-Critic) or in parallel (Q-learning).
