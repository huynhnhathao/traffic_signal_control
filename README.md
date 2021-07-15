# Reinforcement learning methods for traffic signal control problem

This repository contains code to train a Double DQN RL agent for the task of traffic signal control. The data and environment are from the 
[CityBrain challenge 2021](https://kddcup2021-citybrainchallenge.readthedocs.io/en/final-phase/try-it-yourself.html), qualification phase.

## To run the training code:
1. Install docker, pull the docker image of the CBEngine: docker pull citybrainchallenge/cbengine:0.1.2
2. Clone the starter-kit from the challenge: git clone https://github.com/CityBrainChallenge/KDDCup2021-CityBrainChallenge-starter-kit.git, this starter-kit contains the starting code the competition provide.
3. Create a container from the previous image: `docker run -it --name your_container_name --mount type=bind,source=Source\to\your\starter\kit,target=\starter-kit citybrainchallenge/cbengine:0.1.2 bash`
4. To run the code in the container just created, run 3 commands: docker start your_container_name, docker exec -it you_container_name bash, cd starter-kit. Now you are on the starter-kit folder inside the container
5. To run a python file inside the starter-kit folder: python3 filename.py


If your operating system is Windows and you have set `git config --global core.autocrlf true` , when you clone this repo, git will automatically add CR to cfg/simulator.cfg. This will lead to the error in Linux of the docker container.

So please change cfg/simulator.cfg from CRLF to LF after cloning this repo.
