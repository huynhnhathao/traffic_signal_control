# Reinforcement learning methods for traffic signal control problem

This repository contains code to train a Double DQN RL agent for the task of traffic signal control. The data and environment are from the 
[CityBrain challenge 2021](https://kddcup2021-citybrainchallenge.readthedocs.io/en/final-phase/try-it-yourself.html), qualification phase.

## To run the training code:
1. Install docker, pull the docker image of the CBEngine: `docker pull citybrainchallenge/cbengine:0.1.2`
2. Clone the starter-kit from the challenge: `git clone https://github.com/CityBrainChallenge/KDDCup2021-CityBrainChallenge-starter-kit.git`, this starter-kit contains the starting code the competition provide.
3. Create a container from the previous image: `docker run -it --name your_container_name --mount type=bind,source=Source\to\your\starter\kit,target=\starter-kit citybrainchallenge/cbengine:0.1.2 bash`
4. To run the code in the container just created, run 3 commands: `docker start your_container_name`, `docker exec -it you_container_name bash`, `cd starter-kit`. Now you are on the starter-kit folder inside the container
5. To run a python file inside the starter-kit folder: `python3 filename.py`

Now you can test the installation by run the evaluate scripts:

`python3 evaluate.py --input_dir agent --output_dir out --sim_cfg cfg/simulator.cfg --metric_period 200`

if that run succesfully, then you are good to go.

Next, you clone this repository, the 'stater-kit.zip' is like the starter-kit you clone previously. So replace the previous starter-kit with this starter-kit, inside this new starter-kit (after extracted). There are many files, but you just need to focus on some python scripts that I writed: `env_wrapper.py`, `Policies.py`, `MemoryReplay.py`, `Training_DQN.py`, `doubleDQN.py` and the `agent.py` inside the test folder. You can see there are many x.pkl files in the stater-kit, these are the data I extract and transform from the original data to help training easier.

To train the agent, run: python3 Training_DQN.py. This will save the agent periodically, configurable in the Training_DQN.py file.
To test the agent performance, you have to put the policy name saved in the saved_models folder into the agent.py script, inside the test folder. Then run: `python3 evaluate.py --input_dir test --output_dir out --sim_cfg cfg/simulator.cfg --metric_period 200`

My contribution in this work included but not limited to all the files I have just mentioned.

You may can not run the code, and encounter many bugs when installing this. That is normal, You can contact me if there is some issuses when installing/running.

If your operating system is Windows and you have set `git config --global core.autocrlf true` , when you clone this repo, git will automatically add CR to cfg/simulator.cfg. This will lead to the error in Linux of the docker container.

So please change cfg/simulator.cfg from CRLF to LF
