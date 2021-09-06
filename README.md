# BAROCCO
This code accompanies the paper "Balancing Rational and Other-Regarding Preferences in Cooperative-Competitive Environments".

The code is divided into 4 folders, corresponding to the combinations of two envirionments (Harvest and Eldorado) and two frameworks (Q-learning and Actor-Criric).

## Harvest 

Harvest is a popular mixed environment for multi-agent reinforcement learning.

<img src="https://user-images.githubusercontent.com/22059171/125957619-beaa5df5-3534-4576-a49d-4777d06bf5b7.png" height="300">

The code for the environment was provided by [https://github.com/eugenevinitsky/sequential_social_dilemma_games](https://github.com/eugenevinitsky/sequential_social_dilemma_games)

## Eldorado

Eldorado is an original mixed environment based on the NeuralMMO environment ([https://github.com/openai/neural-mmo](https://github.com/openai/neural-mmo)).

<img src="https://user-images.githubusercontent.com/22059171/125957756-435f1dfb-5429-4e8a-9756-8a91d96e5eba.png" height="300">


## Structure

```main.py``` is the main file, run it to train algorithms. Change algorithms, settings, and hyperparameters in ```configs.py```. Make figures from ```logs.py``` running ```figures.py```

The folder ```source``` contains all algorithm-related code. ```networks``` contains all neural networks. ```utils``` contains all optimization stuff required to train the networks. ```env``` contains all environment logic. ```agent``` contains high-level classes: ```learner.py``` and ```learner_social.py``` contain high-level logic to update networks, ```controller.py``` contains high-level logic for agent-environment interaction to collect rollouts, ```runner.py``` alternates between interaction with environment and network updates -- either sequentially (Actor-Critic) or in parallel (Q-learning).


Please use this bibtex if you want to cite this repository in your publications:


    @inbook{10.5555/3463952.3464151,
          author = {Ivanov, Dmitry and Egorov, Vladimir and Shpilman, Aleksei},
          title = {Balancing Rational and Other-Regarding Preferences in Cooperative-Competitive Environments},
          year = {2021},
          isbn = {9781450383073},
          publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
          address = {Richland, SC},
          booktitle = {Proceedings of the 20th International Conference on Autonomous Agents and MultiAgent Systems},
          pages = {1536–1538},
          numpages = {3}
          }
