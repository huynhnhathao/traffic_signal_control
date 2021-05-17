""" Required submission file

In this file, you should implement your `AgentSpec` instance, and **must** name it as `agent_spec`.
As an example, this file offers a standard implementation.
"""
import copy

import os
path = os.path.split(os.path.realpath(__file__))[0]
import sys
sys.path.append(path)
import random

import gym

from pathlib import Path
import gym

import os
from collections import deque
import numpy as np
import logging

import torch
from collections import deque, namedtuple
import random
import torch
import torch.nn as nn 
# import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

gym.logger.setLevel(gym.logger.INFO)

class MemoryReplay(object):
    """Memmory class to save experiences"""
    def __init__(self, maxlen,batch_size, seed, device = 'cpu'):
        """
        params:
            maxlen: max len of memory
            batch_size: number of experiences are sampled each time sample is called
            seed: set seed for random, for reproducible purpose
        """
        self.maxlen = maxlen
        self.seed = seed
        self.memory = deque(maxlen  = self.maxlen)
        self.experience = namedtuple('Experience', field_names= ['state', 'action', 'reward', 'next_state', 'done'])
        self.batch_size = batch_size
        self.device = device
        random.seed(self.seed)

    
    def sample(self, n_experiences = None):
        """
        Sample n_experiences from the memory
        return pytorch tensors of experiences: states, actions, rewards, next_states, dones 
        """
        if n_experiences is None:
            n_experiences = self.batch_size
        
        samples = random.sample(self.memory, n_experiences)
        states, actions, rewards, next_states, dones = np.array([x.state for x in samples]) , np.array([x.action for x in samples]), np.array([x.reward for x in samples]),\
         np.array([x.next_state for x in samples]), np.array([x.done for x in samples])
         
        # logger.info(f'1 example of experience sample from memory: state: {states[0]} action: {actions[0]} reward {rewards[0]} next_state: {next_states[0]} done: {dones[0]}')
        assert type(states) == np.ndarray, 'states is expected to be np.ndarray'
        assert len(states) == len(actions) == len(rewards) == len(next_states) == len(dones), 'len does not match'
        # ndarray into pytorch tensors
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().unsqueeze(-1).to(self.device)
        rewards = torch.from_numpy(rewards).float().unsqueeze(-1).to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().unsqueeze(-1).to(self.device)
        # logger.info(f'Shape of tensor return from memory class: \n states.size() = {states.size()}, actions.size() = {actions.size()}, rewards.size() = {rewards.size()}, next_states.size() = {next_states.size()}, dones.size() = {dones.size()}')
        # is the shape right?
        return states, actions, rewards, next_states, dones

        

    def add_experience(self, experience):
        """
        Add many experiences to the memory
        this add only 1 tuple of (state, action, reward, next_state)
        """
        #num_experiences = len(experiences[0])
        self.memory.extend([self.experience(experience[0], experience[1], experience[2], experience[3], experience[4]) ])
        # logger.info(f'Add {num_experiences} experiences to memory, below are five last added experiences: ')
        # for i in range(5):
        #     experience = self.memory[-i]
        #     logger.info(f"{i}. state: {experience.state} action: {experience.action} reward: {experience.reward} next_state: {experience.next_state} done: {experience.done} ")

    def __len__(self):
        return len(self.memory)


class MLPPolicy(nn.Module):
    def __init__(self, in_dim, out_dim):
        """Create a MLP neural network to be a policy
        parameters: 
            in_dim: int,  input dimension
            out_dim: int, output dimension
        return: logits
        """
        super(MLPPolicy, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.linear1 = nn.Linear(self.in_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64,self.out_dim )
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 0)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.tanh(x)
        x = self.linear3(x)
        return x
            

class DQN(object):
    def __init__(self, memory, model, args):
        """
        parameters:
            memory: instance of MemoryReplay class
            model: instance of a nn class
        """
        self.args = args
        self.memory = memory
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.update_target_network()

        self.now_phase = {}
        self.green_sec = self.args['green_sec']
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}
        #how much experience tuple are needed to start learning, these number are compared to the total_decision_num in training
        self.learning_start = self.args['learning_start']
        self.update_model_freq  = self.args['update_model_freq']
        # after this number of time of num_step_training, the target model will be updated to the model
        self.update_target_model_freq = 5
        self.policy = self.args['policy']
        self.gamma = self.args['gamma']
        # for epsilon greedy
        self.start_epsilon = self.args['start_epsilon']
        self.epsilon = self.args['start_epsilon']
        self.epsilon_min1 = self.args['epsilon_min1']
        self.epsilon_min2 = self.args['epsilon_min2']
        self.epsilon_decay = self.args['epsilon_decay']
        self.learning_rate = self.args['learning_rate']
        
        self.ob_length = 24
        self.action_space = 8

        self.num_batchs_per_step = self.args['num_batchs_per_step']
        self.num_update_per_batch = self.args['num_update_per_batch']
        # track the number of times calling the one_step_training method
        self.num_steps_training = 0
        # after this number of calling the one_step_training_method times, target will be computed using the target net
        self.start_using_target_network = self.args['start_using_target_network']
        self.l2_lambda = self.args['l2_lambda']
        # training stuff

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate )
        self.loss_function = nn.MSELoss()

        #--------------boltzmann policy parameters
        self.boltzmann_start_temperature = self.args['boltzmann_start_temperature']
        self.boltzmann_temperature = self.boltzmann_start_temperature
        self.boltzmann_min_temperature = self.args['boltzmann_min_temperature']
        self.boltzmann_num_decay_steps = self.args['boltzmann_num_decay_steps']
        self.boltzmann_decay_rate = self.compute_boltzmann_decay_rate()

        # the decay ray for 2 phase of epilon greedy policy, phase 1 is from ep 1 -> 20, 
        # with epsilon goes from 0.5 to 0.1, phase 2 from ep 21 -> 70, 
        # with epsilon goes from 0.1 to 0.01
        self.rate1 = (self.epsilon_min1 / self.start_epsilon)**(self.args['num_decay_epsilon_phase_1'])
        self.rate2 = (self.epsilon_min2 / self.epsilon_min1)**(self.args['num_decay_epsilon_phase_2'])
    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self,agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list,1)
        self.last_change_step = dict.fromkeys(self.agent_list,0)
    # intersections[key_id] = {
    #     'have_signal': bool,
    #     'end_roads': list of road_id. Roads that end at this intersection. The order is random.
    #     'start_roads': list of road_id. Roads that start at this intersection. The order is random.
    #     'lanes': list, contains the lane_id in. The order is explained in Docs.
    # }
    # roads[road_id] = {
    #     'start_inter':int. Start intersection_id.
    #     'end_inter':int. End intersection_id.
    #     'length': float. Road length.
    #     'speed_limit': float. Road speed limit.
    #     'num_lanes': int. Number of lanes in this road.
    #     'inverse_road':  Road_id of inverse_road.
    #     'lanes': dict. roads[road_id]['lanes'][lane_id] = list of 3 int value. Contains the Steerability of lanes.
    #               lane_id is road_id*100 + 0/1/2... For example, if road 9 have 3 lanes, then their id are 900, 901, 902
    # }
    # agents[agent_id] = list of length 8. contains the inroad0_id, inroad1_id, inroad2_id,inroad3_id, outroad0_id, outroad1_id, outroad2_id, outroad3_id
    # each agent has control over 8 roads, which made up one intersection(eight is the most, there maybe has 3 legs intersection which has only 6 roads)

    def load_roadnet(self,intersections, roads, agents):
        self.intersections = intersections
        self.roads = roads
        self.agents = agents
    ################################
    


    def act_(self, observations_for_agent, ):
        #This function return action for training, using epsilon greedy or boltzmann policy
        actions = {}
        with torch.no_grad():
            for agent_id in self.agent_list:
                action = self.get_action(observations_for_agent[agent_id])
                actions[agent_id] = action
        return actions

    def act(self, obs):
        # this function is for inferences, use greedy policy 100%
        observations = obs['observations']
        #info = obs['info']
        actions = {}

        # Get state
        observations_for_agent = {}
        for key,val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key[key.find('_')+1:]
            if(observations_agent_id not in observations_for_agent.keys()):
                observations_for_agent[observations_agent_id] = {}
            observations_for_agent[observations_agent_id][observations_feature] = val[1:]

        # Get actions
        # inference do not need gradient
        with torch.no_grad():
            for agent in self.agent_list:
                # plus 1 because action range from 1 to 8, not 0 to 7
                q_values = self.model(observations_for_agent[agent]['lane_vehicle_num'] )
                actions[agent] = torch.argmax(q_values).item() + 1
                #actions[agent] = self.get_action(observations_for_agent[agent]['lane_vehicle_num']) + 1  

        return actions

    def get_action(self, ob, ):

        # The epsilon-greedy action selector.
   
        ob = self._reshape_observation(ob)
        q_values = self.model(ob.float())
        if self.policy == "boltzmann":
            action = self.boltzmann_policy(q_values)
            return action
        elif self.policy == 'epsilon_greedy':
            action = self.epsilon_greedy(q_values)
            return action
        

    def sample(self):
        # randomly sample action
        return np.random.randint(0, self.action_space)

    def _reshape_observation(self, observation):
        # 1 row
        # this should be a pytorch tensor
        # we change it later
        try:
            return torch.from_numpy(np.reshape(observation['lane'], (1, -1)), )
        except:
            return torch.from_numpy(np.reshape(observation, (1, -1)), )

    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(copy.deepcopy(weights))
        # This function has tested and worked properly 
        logger.info('Udated target net')
        # logger.info(f' target model weight { self.target_model.state_dict()}' )
    
        # logger.info(f' model state dict {self.model.state_dict()} ')

    def load_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.pt".format(step)
        model_name = os.path.join(dir, name)
        print("load from " + model_name)
        self.model.load_state_dict(torch.load(model_name))

    def save_model(self, dir="model/dqn", ):
        name = "dqn_agent_{}.pt".format(self.num_steps_training)
        model_name = os.path.join(dir, name)    
        torch.save(self.model.state_dict(), model_name)

    # def train_network(self):
    #     """Training this self.model"""
    #     if self.memory.batch_s
    
    def calculate_target(self,  rewards, next_states, dones, using_target_net):
        """
        Calculate the Q_tar for each example in batch
        states, actions, rewards, next_states, dones are expected to be pytorch tensors
        """
        targets = []
        # deactivate batch norm, dropout
        # no need to compute gradients

        # logger.info(f'expected tensor size in calculated_target: [ {self.batch_size}, state_size]')
        # logger.info(f'tensor size received in calculate_target: states.size() = {states.size()}, actions.size() = {actions.size()}, rewards.size() = {rewards.size()}, next_states.size() = {next_states.size()}, dones.size() = {dones.size()}')
        # next_q_values = []
        
        for reward, next_state, done in zip(rewards, next_states, dones):

            max_next_q_value = self.compute_max_next_q_value(next_state, using_target_net)
            # next_q_values.append(max_next_q_value)
            q_value = reward + self.gamma*max_next_q_value*int(done)
            targets.append(q_value)
        # print(f'Max next q values {next_q_values}')
        # will the shape correct?
        # targets has size [batch_size]

        ######## ISSUE HERE: max_next_q_value keep increasing as training progress, cause exploding reward
        return torch.tensor(targets, dtype= torch.float).unsqueeze(-1)################################################# Check shape return

    def compute_max_next_q_value(self, next_state, using_target_net):
        """compute the maximum q value of next state"""
        q_values =  self.target_model(next_state)# if using_target_net else self.model(next_state) if not using_target_net  
        result = torch.max(q_values).item()  
        # logger.info(f'in compute max next q_values, q_values {q_values}, result {result}')
        return   result

    def predict_q_values(self, states, actions):
        """
        Predict q values of all action
        parameters: 
            states: tensor of shape [batch_size, state_size]
            actions: tensor of shape[batch_size]
        return a tensor of shape [batch_size] of predicted q value for each sample
        this time we need gradient
        """
        predicted = self.model(states).gather(1, actions.long())
        # logger.info(f'Q value predicted {predicted}')
        # writer.add_graph(self.model, states)
        return predicted
    



    def L2_weight_regularization(self, l2_lambda):
        """Return the L2 regularization of the policy weights"""
        l2_reg = torch.tensor(0.)
        for param in self.model.parameters():
            l2_reg += torch.norm(param)
        return l2_reg*l2_lambda



#----------------------------------------------------------------------------------------------------------------------
    # boltzmann policy here
    def boltzmann_policy(self, q_values,clips = [-100, 100] ):
        """
        Boltzmann policy for chossing an action

        """
        exp_divide_temperature = torch.exp(torch.clip(q_values/self.boltzmann_temperature,clips[0], clips[1] ))
        # sum = torch.sum(exp_divide_temperature)
        # results = exp_divide_temperature/sum
        try:
            sum_value = torch.sum(exp_divide_temperature)
            d = torch.distributions.categorical.Categorical(logits = exp_divide_temperature/sum_value)
        except:
            logger.error(f'Error at boltzmann_policy')
            print(exp_divide_temperature)
            print(self.boltzmann_temperature)
            print(q_values)

        # d = torch.distributions.categorical.Categorical(logits = exp_divide_temperature)
        return d.sample().item()

    def compute_boltzmann_decay_rate(self):
        """ compute the decay rate for boltzmann policy, 
        from start temperature to min temperature in num time steps"""

        assert self.boltzmann_start_temperature is not None, 'start temperature must be specified'
        assert self.boltzmann_min_temperature is not None, 'min temperature must be specified'
        assert self.boltzmann_num_decay_steps is not None, 'num decay steps must be specified'

        decay_rate = np.log(self.boltzmann_min_temperature/self.boltzmann_start_temperature)/self.boltzmann_num_decay_steps
        logger.info(f"Computed boltzmann decay rate = {decay_rate}")
        return decay_rate   

    def update_boltzmann_temperature(self, episode_num):
        """ Update the boltzmann temperature each global time step"""
        if self.boltzmann_temperature <= self.boltzmann_min_temperature:
            self.boltzmann_temperature = self.boltzmann_min_temperature
            return
        self.boltzmann_temperature = self.boltzmann_start_temperature*np.exp(self.boltzmann_decay_rate*episode_num)

    #---------------------------------------------------------------------------------------------------------------------------
    # Epsilon greedy here
    def epsilon_greedy(self, q_values):
        """
        Epsilon greedy policy
        """
        if not hasattr(self, 'epsilon') and hasattr(self, 'epsilon_min'):
            raise AttributeError("epsilon not specified")

        # q_values is a torch tensor
        #q_values = q_values.view(1, -1)
        if np.random.rand() < self.start_epsilon:
            action = self.sample()
        else:
            action = torch.argmax(q_values, ).item()
        return action

        
    def update_epsilon(self, ):
        """
        this function update epsilon of epsilon_greedy, as follow:
        from episode 1 to episode 20, epsilon goes from 0.5 to 0.1
        from episode 21 to 70, epsilon goes from 0.1 to 0.01
        from episode 70 to 100, epsilon stay at 70
        epsilon_greedy policy only used when training, when inference, use greedy policy
        """
        # rate1 is used for first phase from episode 1 to 20

        if self.epsilon > self.epsilon_min1:
            self.epsilon*= rate
        self.epsilon = self.epsilon_decay*self.epsilon

    def one_step_training(self, ):
        """
        Do one step training model here
        For some first update, the target net should be the original model
        """
        self.model.train()
        self.model.eval()
        this_step_loss = 0
        using_target_net = False if self.num_steps_training < self.start_using_target_network else True
        for i in range(self.num_batchs_per_step):
            # sample a batch from memory
            states, actions, rewards, next_states, dones = self.memory.sample()
            # logger.info(f'inside one batch sampling, states {states} \n, actions {actions} \n, rewards {rewards} \n, next_states {next_states}\n , dones {dones}\n')
            # logger.info(f'Rewards {rewards}')
            for u in range(self.num_update_per_batch):
                # calculate target, loss, do an update
                # logger.info(f'inside one update')
                self.optimizer.zero_grad()
                #logger.info('after zerograd')
                self.model.eval()
                with torch.no_grad():
                    targets = self.calculate_target( rewards, next_states, dones, using_target_net)
                self.model.train()
                #logger.info(f'target computed {targets}')
                preds = self.predict_q_values(states, actions)
                #logger.info(f'predicted q values {preds}')
                # logger.info(f'target shape {targets.size()} predicted shape {preds.size()}') This ok [64, 1]

                loss = self.loss_function(preds, targets)
                loss += self.L2_weight_regularization(self.l2_lambda)
                #logger.info(f'after zero grad, before call backward {self.model.linear1.weight.grad}')
                loss.backward()
                #logger.info(f'after backward,  {self.model.linear1.weight.grad}')
                #norms = torch.nn.utils.clip_grad_norm_(self.model.parameters(),100 )
                #logger.info(f'after clip norm linear1 {self.model.linear1.weight.grad} \n linear2{self.model.linear2.weight.grad} \n linear3{self.model.linear3.weight.grad} \n with norm {norms}')
                self.optimizer.step()
                this_step_loss = loss.detach().item()
                
        self.num_steps_training +=1
        #self.update_boltzmann_temperature()
        logger.info(f'Called one_step_training {self.num_steps_training}, loss = {this_step_loss}')


        
scenario_dirs = ["test" ]


memory = MemoryReplay(2000, 64, 42,)
net = MLPPolicy(24, 8)
agent_specs = dict.fromkeys(scenario_dirs, None)
args = {
    'green_sec': 40, 
    # how many episode should be completed for learning to start
    'learning_start':  1, 
    # how many episode between each time calling the one_step_training
    'update_model_freq': 4, 
    'update_target_model_freq': 5, 
    # after this number of time of num_step_training, the target model will be updated to the model
    'gamma': 0.95, 
    'start_epsilon': 0.5, 
    'epsilon_min1' : 0.1, 
    'epsilon_min2': 0.01, 
    'epsilon_decay': None,

    'learning_rate': 0.005, 
    'num_batchs_per_step': 30, 
    'num_update_per_batch': 2, 
    'start_using_target_network': 10, 
    'l2_lambda': 0.01, 

    'boltzmann_start_temperature': 5, 
    'boltzmann_min_temperature': 1,
    # boltzmann temperature updated after each episode
    'boltzmann_num_decay_steps': 30 ,
    "policy": 'epsilon', 
    "num_decay_epsilon_phase_1": 20, 
    "num_decay_epsilon_phase_2": 40
    # how many training step before start using target net

}
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = DQN(memory,net ,args)
