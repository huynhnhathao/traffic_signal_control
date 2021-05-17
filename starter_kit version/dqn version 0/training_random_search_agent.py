import datetime
import CBEngine
import json
import traceback
import argparse
import logging
import os
import sys
import time
from pathlib import Path
import re
import gym
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

data_logger = logging.getLogger(__file__)
data_logger.setLevel(logging.ERROR)

gym.logger.setLevel(gym.logger.INFO)

#region
def pretty_files(path):
    contents = os.listdir(path)
    return "[{}]".format(", ".join(contents))


def resolve_dirs(root_path: str, input_dir: str = None, output_dir: str = None):
    root_path = Path(root_path)

    logger.info(f"root_path={pretty_files(root_path)}")

    if input_dir is not None:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        submission_dir = input_dir
        scores_dir = output_dir

        logger.info(f"input_dir={pretty_files(input_dir)}")
        logger.info(f"output_dir={pretty_files(output_dir)}")
    else:
        raise ValueError('need input dir')

    if not scores_dir.exists():
        os.makedirs(scores_dir)

    logger.info(f"submission_dir={pretty_files(submission_dir)}")
    logger.info(f"scores_dir={pretty_files(scores_dir)}")

    if not submission_dir.is_dir():
        logger.warning(f"submission_dir={submission_dir} does not exist")

    return submission_dir, scores_dir


def load_agent_submission(submission_dir: Path):
    logger.info(f"files under submission dir:{pretty_files(submission_dir)}")

    # find agent.py
    module_path = None
    cfg_path = None
    for dirpath, dirnames, file_names in os.walk(submission_dir):
        for file_name in [f for f in file_names if f.endswith(".py")]:
            if file_name == "agent.py":
                module_path = dirpath

            if file_name == "gym_cfg.py":
                cfg_path = dirpath
    # error
    assert (
        module_path is not None
    ), "Cannot find file named agent.py, please check your submission zip"
    assert(
        cfg_path is not None
    ), "Cannot find file named gym_cfg.py, please check your submission zip"
    sys.path.append(str(module_path))


    # This will fail w/ an import error of the submissions directory does not exist
    import gym_cfg as gym_cfg_submission
    import random_agent as agent_submission

    gym_cfg_instance = gym_cfg_submission.gym_cfg()

    return  agent_submission.agent_specs,gym_cfg_instance


def read_config(cfg_file):
    configs = {}
    with open(cfg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n').split(' ')
            if(len(line) == 3 and line[0][0] != '#'):
                configs[line[0]] = line[-1]
    return configs


def process_roadnet(roadnet_file):
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

    intersections = {}
    roads = {}
    agents = {}

    agent_num = 0
    road_num = 0
    signal_num = 0
    with open(roadnet_file, 'r') as f:
        lines = f.readlines()
        cnt = 0
        pre_road = 0
        is_obverse = 0
        for line in lines:
            line = line.rstrip('\n').split(' ')
            if ('' in line):
                line.remove('')
            if (len(line) == 1):
                if (cnt == 0):
                    agent_num = int(line[0])
                    cnt += 1
                elif (cnt == 1):
                    road_num = int(line[0]) * 2
                    cnt += 1
                elif (cnt == 2):
                    signal_num = int(line[0])
                    cnt += 1
            else:
                if (cnt == 1):
                    intersections[int(line[2])] = {
                        'have_signal': int(line[3]),
                        'end_roads': [],
                        'start_roads': [],
                        'lanes':[]
                    }
                elif (cnt == 2):
                    if (len(line) != 8):
                        road_id = pre_road[is_obverse]
                        roads[road_id]['lanes'] = {}
                        for i in range(roads[road_id]['num_lanes']):
                            roads[road_id]['lanes'][road_id * 100 + i] = list(map(int, line[i * 3:i * 3 + 3]))
                        is_obverse ^= 1
                    else:
                        roads[int(line[-2])] = {
                            'start_inter': int(line[0]),
                            'end_inter': int(line[1]),
                            'length': float(line[2]),
                            'speed_limit': float(line[3]),
                            'num_lanes': int(line[4]),
                            'inverse_road': int(line[-1])
                        }
                        roads[int(line[-1])] = {
                            'start_inter': int(line[1]),
                            'end_inter': int(line[0]),
                            'length': float(line[2]),
                            'speed_limit': float(line[3]),
                            'num_lanes': int(line[5]),
                            'inverse_road': int(line[-2])
                        }
                        intersections[int(line[0])]['end_roads'].append(int(line[-1]))
                        intersections[int(line[1])]['end_roads'].append(int(line[-2]))
                        intersections[int(line[0])]['start_roads'].append(int(line[-2]))
                        intersections[int(line[1])]['start_roads'].append(int(line[-1]))
                        pre_road = (int(line[-2]), int(line[-1]))
                else:
                    # 4 out-roads
                    signal_road_order = list(map(int, line[1:]))
                    now_agent = int(line[0])
                    in_roads = []
                    for road in signal_road_order:
                        if (road != -1):
                            in_roads.append(roads[road]['inverse_road'])
                        else:
                            in_roads.append(-1)
                    in_roads += signal_road_order
                    agents[now_agent] = in_roads
    for agent, agent_roads in agents.items():
        intersections[agent]['lanes'] = []
        for road in agent_roads:
            ## here we treat road -1 have 3 lanes
            if (road == -1):
                for i in range(3):
                    intersections[agent]['lanes'].append(-1)
            else:
                for lane in roads[road]['lanes'].keys():
                    intersections[agent]['lanes'].append(lane)

    return intersections, roads, agents


def process_delay_index(lines, roads, step):
    vehicles = {}

    for i in range(len(lines)):
        line = lines[i]
        if(line[0] == 'for'):
            vehicle_id = int(line[2])
            now_dict = {
                'distance': float(lines[i + 1][2]),
                'drivable': int(float(lines[i + 2][2])),
                'road': int(float(lines[i + 3][2])),
                'route': list(map(int, list(map(float, lines[i + 4][2:])))),
                'speed': float(lines[i + 5][2]),
                'start_time': float(lines[i + 6][2]),
                't_ff': float(lines[i+7][2]),
            ##############
                'step': int(lines[i+8][2])
            }
            step = now_dict['step']
            ##################
            vehicles[vehicle_id] = now_dict
            tt = step - now_dict['start_time']
            tt_ff = now_dict['t_ff']
            tt_f_r = 0.0
            current_road_pos = 0
            for pos in range(len(now_dict['route'])):
                if(now_dict['road'] == now_dict['route'][pos]):
                    current_road_pos = pos
            for pos in range(len(now_dict['route'])):
                road_id = now_dict['route'][pos]
                if(pos == current_road_pos):
                    tt_f_r += (roads[road_id]['length'] -
                               now_dict['distance']) / roads[road_id]['speed_limit']
                elif(pos > current_road_pos):
                    tt_f_r += roads[road_id]['length'] / roads[road_id]['speed_limit']
            vehicles[vehicle_id]['tt_f_r'] = tt_f_r
            vehicles[vehicle_id]['delay_index'] = (tt + tt_f_r) / tt_ff

    vehicle_list = list(vehicles.keys())
    delay_index_list = []
    for vehicle_id, dict in vehicles.items():
        # res = max(res, dict['delay_index'])
        if('delay_index' in dict.keys()):
            delay_index_list.append(dict['delay_index'])

    # 'delay_index_list' contains all vehicles' delayindex at this snapshot.
    # 'vehicle_list' contains the vehicle_id at this snapshot.
    # 'vehicles' is a dict contains vehicle infomation at this snapshot
    return delay_index_list, vehicle_list, vehicles

def process_score(log_path,roads,step,scores_dir):
    result_write = {
        "data": {
            "total_served_vehicles": -1,
            "delay_index": -1
        }
    }

    with open(log_path / "info_step {}.log".format(step)) as log_file:
        lines = log_file.readlines()
        lines = list(map(lambda x: x.rstrip('\n').split(' '), lines))
        # process delay index
        delay_index_list, vehicle_list, vehicles = process_delay_index(lines, roads, step)
        v_len = len(vehicle_list)
        delay_index = np.mean(delay_index_list)

        result_write['data']['total_served_vehicles'] = v_len
        result_write['data']['delay_index'] = delay_index
        with open(scores_dir / 'scores {}.json'.format(step), 'w' ) as f_out:
            json.dump(result_write,f_out,indent= 2)

    return result_write['data']['total_served_vehicles'],result_write['data']['delay_index']

# this is for evaluate agent performance

def run_simulation(agent_spec, simulator_cfg_file, gym_cfg,metric_period,scores_dir,threshold):
    logger.info("\n")
    logger.info("*" * 40)

    # get gym instance
    gym_configs = gym_cfg.cfg
    simulator_configs = read_config(simulator_cfg_file)
    env = gym.make(
        'CBEngine-v0',
        simulator_cfg_file=simulator_cfg_file,
        thread_num=1,
        gym_dict = gym_configs,
        metric_period = metric_period
    )
    scenario = [
        'test'
    ]

    # read roadnet file, get data
    roadnet_path = Path(simulator_configs['road_file_addr'])
    intersections, roads, agents = process_roadnet(roadnet_path)
    env.set_warning(1)
    env.set_log(1)
    env.set_info(1)
    env.set_ui(1)
    # get agent instance
    observations, infos = env.reset()
    agent_id_list = []
    for k in observations:
        agent_id_list.append(int(k.split('_')[0]))
    agent_id_list = list(set(agent_id_list))
    agent = agent_spec[scenario[0]]
    agent.load_agent_list(agent_id_list)
    agent.load_roadnet(intersections, roads, agents)
    done = False
    # simulation
    step = 0
    log_path = Path(simulator_configs['report_log_addr'])
    sim_start = time.time()

    tot_v  = -1
    d_i = -1
    while not done:
        actions = {}
        step+=1
        all_info = {
            'observations':observations,
            'info':infos
        }
        actions = agent.act(all_info)
        observations, rewards, dones, infos = env.step(actions)
        if(step * 10 % metric_period == 0):
            try:
                tot_v , d_i = process_score(log_path,roads,step*10-1,scores_dir)
            except Exception as e:
                print(e)
                print('Error in process_score. Maybe no log')
                continue
        if(d_i > threshold):
            break
        for agent_id in agent_id_list:
            if(dones[agent_id]):
                done = True
    sim_end = time.time()
    logger.info("simulation cost : {}s".format(sim_end-sim_start))
    # read log file

    # result = {}
    # vehicle_last_occur = {}

    # eval_start = time.time()
    # for dirpath, dirnames, file_names in os.walk(log_path):
    #     for file_name in [f for f in file_names if f.endswith(".log") and f.startswith('info_step')]:
    #         with open(log_path / file_name, 'r') as log_file:
    #             pattern = '[0-9]+'
    #             step = list(map(int, re.findall(pattern, file_name)))[0]
    #             if(step >= int(simulator_configs['max_time_epoch'])):
    #                 continue
    #             lines = log_file.readlines()
    #             lines = list(map(lambda x: x.rstrip('\n').split(' '), lines))
    #             result[step] = {}
    #             # result[step]['vehicle_num'] = int(lines[0][0])
    #
    #             # process delay index
    #             delay_index_list, vehicle_list, vehicles = process_delay_index(lines, roads, step)
    #             result[step]['vehicle_list'] = vehicle_list
    #             result[step]['delay_index'] = delay_index_list
    #             result[step]['vehicles'] = vehicles
    #
    #
    # steps = list(result.keys())
    # steps.sort()
    # for step in steps:
    #     for vehicle in result[step]['vehicles'].keys():
    #         vehicle_last_occur[vehicle] = result[step]['vehicles'][vehicle]
    #
    # delay_index_temp = {}
    # for vehicle in vehicle_last_occur.keys():
    #     if('delay_index' in vehicle_last_occur[vehicle]):
    #         res = vehicle_last_occur[vehicle]['delay_index']
    #         delay_index_temp[vehicle] = res
    #
    # # calc
    # vehicle_total_set = set()
    # delay_index = []
    # for k, v in result.items():
    #     vehicle_total_set = vehicle_total_set | set(v['vehicle_list'])
    #     delay_index += delay_index_list
    #
    # if(len(delay_index)>0):
    #     d_i = np.mean(delay_index)
    # else:
    #     d_i = -1
    #
    # last_d_i = np.mean(list(delay_index_temp.values()))
    # eval_end = time.time()
    # logger.info("scoring cost {}s".format(eval_end-eval_start))
    return tot_v,  d_i


def format_exception(grep_word):
    exception_list = traceback.format_stack()
    exception_list = exception_list[:-2]
    exception_list.extend(traceback.format_tb(sys.exc_info()[2]))
    exception_list.extend(traceback.format_exception_only(
        sys.exc_info()[0], sys.exc_info()[1]))
    filtered = []
    for m in exception_list:
        if str(grep_word) in m:
            filtered.append(m)

    exception_str = "Traceback (most recent call last):\n"
    exception_str += "".join(filtered)
    # Removing the last \n
    exception_str = exception_str[:-1]

    return exception_str






# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Here is my real code

class Trainer(object):
    """Train an agent with random search"""
    def __init__(self, args, agent_specs, simulator_cfg_file, gym_cfg, metric_period):
    

        self.args = args


        
        self.agent_spec = agent_spec
        self.simulator_cfg_file = simulator_cfg_file
        self.gym_cfg = gym_cfg
        self.metric_period = metric_period
        self.simulator_configs = read_config(self.simulator_cfg_file)
        self.gym_configs = gym_cfg.cfg
        self.env = gym.make('CBEngine-v0', simulator_cfg_file = self.simulator_cfg_file,
                            thread_num = 1,
                            gym_dict = self.gym_configs,
                            metric_period = self.metric_period 
                            )
                            
        self.roadnet_path = Path(self.simulator_configs['road_file_addr'])
        self.intersections, self.roads, self.agents = process_roadnet(self.roadnet_path)
        self.env.set_log(0)
        self.env.set_warning(0)
        self.env.set_ui(0)
        self.env.set_info(0)

        self.scenario = ["test"]
        #agent is the agent instance
        self.agent = agent_specs[self.scenario[0]]

        
    def get_agent_id_list(self, observations):
        """Get the agent_id_list from observation and return it"""
        agent_id_list = []
        for k in observations:

            agent_id_list.append(int(k.split('_')[0]))
        agent_id_list = list(set(agent_id_list))
        return agent_id_list

    def get_observations_for_agent(self, observations):
        """
        split observation for each intersection agent
        observation received is the observation return from env.reset() or env.step(actions)
        """
        observations_for_agent = {}
        for key, val in observations.items():
            observation_agent_id = int(key.split('_')[0])# agent id
            observation_feature = key.split('_')[1] # lane?
            if observation_agent_id not in observations_for_agent:
                observations_for_agent[observation_agent_id] = {}

            val  = val[1:] # leave out the time step
            while len(val) < self.agent.ob_length:
                val.append(0)
            observations_for_agent[observation_agent_id][observation_feature] = val
            # this is only take the lane_vehicle_num to each agent's observation

        return observations_for_agent
    

    def transform_action(self, actions):
        """This take an action dict, plus all action to 1"""
        actions_ = {}
        for key in actions.keys():
            actions_[key] = actions[key] + 1
        return actions_

    def run_one_episode(self):
        """Run 1 episode in the environment, add experiences to memory"""



    def train(self, ):

        logger.info("\n")
        logger.info("*" * 40)

        observations, infos = self.env.reset()

        self.agent_id_list  = self.get_agent_id_list(observations)
        # I should know what is agent_spec
        # here agent is just one policy
        # load_agent_list save all agent ids into agent object and create 2 dict: now_phase and last_change_step
        self.agent.load_agent_list(self.agent_id_list)
        # load_roadnet save the intersections, roads, agents into agent object
        self.agent.load_roadnet(self.intersections,self.roads,self.agents)
        # Here begins the code for training
        
        total_decision_num = 0

        # agent.load_model(args.save_dir, 199)

        # The main loop
        for e in range(self.args.episodes):
            print("--------------------------------------------------------------------------------------------------------- Episode {}/{}".format(e, self.args.episodes))
            last_obs = self.env.reset()
            # every agent has it own rewards save in episodes_rewards
            episodes_rewards = {}
            for agent_id in self.agent_id_list:
                episodes_rewards[agent_id] = 0
            episodes_decision_num = 0

            # Begins one simulation.
            # i is environment step, or second?
            i = 0
            # number of step each episode?
            # step here is set to 360,
            # it is the number of time call env.step()
            
            while i < self.args.steps:  #360
                if i %120 == 0:
                    print(f'Episode {e}, step {i}' + '-'*100)
                # if the previous phase has completed
                if i % self.args.action_interval == 0:
                    if isinstance(last_obs, tuple):
                        observations = last_obs[0]
                    else:
                        observations = last_obs

                    data_logger.info(f'Last obs is {last_obs}')
                    data_logger.info(f'observations is {observations}')
                    # action's keys will be the itersection ids, and its action will be the phase
                    actions = {}
                    # Get the state.
                    # to has action for each intersection, we need obs for each intersection
                    observations_for_agent = self.get_observations_for_agent(observations)
                    data_logger.info(f'Observation for agent is {observations_for_agent}')
                    # from here we have completed the observations_for_agent dict
                    # Get the action, note that we use act_() for training.
                    # actions should return all the actions of every intersection, or agent
                    actions = self.agent.act_(observations_for_agent)
                    data_logger.info(f'actions before transform {actions}')


                    # plus 1 because environment take action from [1 to 8]
                    actions_ = self.transform_action(actions)
                    data_logger.info(f'actions after transform {actions_}')
                    # We keep the same action for a certain time
                    # because each action is happened in action_interval seconds
                    # you have to call env.step() 4 times with the same actions, 
                    # each time equal 10 sec in env, so it means 40 sec in in the env
                    rewards_list = {}
                    for _ in range(self.args.action_interval):
                        # when in ther interval of one action, we just take the old actions
                        # print(i)
                        data_logger.info('In this interval, we call the same set of action')
                        i += 1
                        # Interacts with the environment and get the reward.
                        # rewards are calculated as 
                        observations, rewards, dones, infos = self.env.step(actions_)
                        data_logger.info('one step just called to the env, 10 sec passed')
                        # for each intersection, reward is the difference between total vehicles on exitting lanes 
                        # and total vehicle on entering lanes, normalized by action_interval, which is the time of each phase
                        # this formula of reward mean that: reward is the number of car go from entering to exitting lanes of one intersection for each second,
                        # if after one phase, the number of cars on entering lanes is more than the number of cars on exitting lanes then rewards is negative
                        
                        # Here 
                        
                        for agent_id in self.agent_id_list:
                            lane_vehicle = observations["{}_lane_vehicle_num".format(agent_id)]
                            pressure = (np.sum(lane_vehicle[13: 25]) - np.sum(lane_vehicle[1: 13])) / self.args.action_interval
                            if agent_id in rewards_list:
                                rewards_list[agent_id] += pressure
                            else:
                                rewards_list[agent_id] = pressure
                    # logger.info(f'reward list for each agent in episode {e} step {i}: {}')

                    data_logger.info(f'conplete one phase, reward for each agent are {rewards_list}')
                    
                    rewards = rewards_list

                    # Get next state.
                    new_observations_for_agent = self.get_observations_for_agent(observations)
                    data_logger.info(f'just get new observation for agent for next decision, {new_observations_for_agent}')
                    """
                    lane here is the number of vehicle on lane
                    observations:
                    {14670355735: {'lane': [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1]}, .....}

                    actions:
                     {42167350403: 3, 22372852612: 6, 42167350405: 1, 42294477575: 1,
                     42294463892: 6, 42167350420: 3, 14670355735: 1, 42381408549: 1, 14790105773: 3,
                      42167350445: 1, 42365406897: 1, 44052836274: 1, 12365406899: 6, 42167350456: 1,
                       22318148289: 6, 22318148293: 1, 42365409354: 1, 42167350476: 1, 42167350479: 1, 
                       13987210067: 1, 42265990106: 3, 14454920703: 1}

                    rewards: 
                    {42167350403: 0.0, 22372852612: -0.25, 42167350405: 0.25, 42294477575: 0.0, 
                    42294463892: 0.0, 42167350420: 0.0, 14670355735: 0.0, 42381408549: 0.0, 14790105773: 0.0,
                     42167350445: 0.0, 42365406897: 0.0, 44052836274: 0.0, 12365406899: 0.0, 42167350456: 0.0,
                      22318148289: 0.0, 22318148293: 0.0, 42365409354: 0.0, 42167350476: 0.0, 42167350479: 0.0,
                       13987210067: 0.0, 42265990106: 0.0, 14454920703: 0.0}

                    new_observations similar as observations
                    """
                    # Remember (state, action, reward, next_state) into memory buffer.
                    for agent_id in self.agent_id_list:
                        self.agent.memory.add_experience((observations_for_agent[agent_id]['lane'], actions[agent_id], rewards[agent_id],
                                    new_observations_for_agent[agent_id]['lane'], dones[agent_id]))
                        data_logger.info(f"add one experience to memory {observations_for_agent[agent_id]['lane']} , {actions[agent_id]}, {rewards[agent_id]}, {new_observations_for_agent[agent_id]['lane']}, {dones[agent_id]}")
                                   
                    # if total_decision_num % 100 == 0:
                    #     logger.info(f'Total decision number so far {}')
                    #     # print(self.agent.memory.memory[-1])

                    
                        episodes_rewards[agent_id] += rewards[agent_id] 

                    
                    episodes_decision_num += 1
                    # count the number of experiences
                    total_decision_num += 1

                    last_obs = observations
                

                # Here each intersection has completed its one phase
                # Update the network
                # if enough experiences, then do an update
                # learning start is the number of experience required to start learning

                if e > self.agent.learning_start and total_decision_num % self.agent.update_model_freq == 0:
                    self.agent.one_step_training()

                if self.agent.num_steps_training+1  % self.agent.update_target_model_freq == 0 and self.agent.num_steps_training > self.agent.start_using_target_network:
                #     print(self.agent.num_steps_training)
                #     print(self.agent.update_target_model_freq)
                    self.agent.update_target_network()
                # if every intersection is done, which is determined by the environment, then stop this episode
                if all(dones.values()):
                    break
            print(f'Episode {e} end with episode_decision_number {episodes_decision_num}')
            print(f'Total decision number so far {total_decision_num}')
            # out of training loop
            if e % self.args.save_rate == self.args.save_rate - 1:
                if not os.path.exists(self.args.save_dir):
                    os.makedirs(self.args.save_dir)
                self.agent.save_model(self.args.save_dir)
            logger.info(
                "episode:{}/{}, average travel time:{}".format(e, self.args.episodes, self.env.eng.get_average_travel_time()))
            for agent_id in self.agent_id_list:
                logger.info(
                    "agent:{}, mean_episode_reward:{}".format(agent_id,
                                                            episodes_rewards[agent_id] / episodes_decision_num))

            self.agent.update_boltzmann_temperature(e)

if __name__ == "__main__":
    # arg parse
    print('*'*500)
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description="1"
    )

    parser.add_argument(
        "--input_dir",
        help="The path to the directory containing the reference "
             "data and user submission data.",
        default= 'agent',
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        help="The path to the directory where the submission's "
             "scores.txt file will be written to.",
        default='out',
        type=str,
    )

    parser.add_argument(
        "--sim_cfg",
        help='The path to the simulator cfg',
        default='cfg/simulator.cfg',
        type=str
    )

    parser.add_argument(
        "--metric_period",
        help="period of scoring",
        default=3600,
        type=int
    )
    parser.add_argument(
        "--threshold",
        help="period of scoring",
        default=1.6,
        type=float
    )

    parser.add_argument('--thread', type=int, default=8, help='number of threads')
    parser.add_argument('--steps', type=int, default=360, help='number of steps')
    # because each time call env.step(), 10 sec passed, so 4 here mean call env.step() 4 time, 40 sec passed
    # in the env, and 4 sec of redlight
    parser.add_argument('--action_interval', type=int, default=3, help='how often agent make decisions')
    parser.add_argument('--episodes', type=int, default=100 , help='training episodes')

    parser.add_argument('--save_model', action="store_true", default=False)
    parser.add_argument('--load_model', action="store_true", default=False)
    parser.add_argument("--save_rate", type=int, default=10,
                        help="save model once every time this many episodes are completed")
    parser.add_argument('--save_dir', type=str, default="model/dqn_warm_up",
                        help='directory in which model should be saved')
    parser.add_argument('--log_dir', type=str, default="cmd_log/dqn_warm_up", help='directory in which logs should be saved')

    # result to be written in out/result.json
    result = {
        "success": False,
        "error_msg": "",
        "data": {
            "total_served_vehicles": -1,
            "delay_index": -1
        }
    }

    args = parser.parse_args()
    # args = edict({
    #     'action_interval'=2, 
    #     'episodes'=100,
    #     'input_dir'='agent',
    #     'load_model'=False,
    #     'log_dir'='cmd_log/dqn_warm_up',
    #     'metric_period'=3600, output_dir='out',
    #     'save_dir'='model/dqn_warm_up', save_model=False,
    #     'save_rate'=5, sim_cfg='cfg/simulator.cfg',
    #     steps=360, thread=8, threshold=1.6
    # })


    msg = None
    metric_period = args.metric_period
    threshold = args.threshold
    # get input and output directory
    simulator_cfg_file = args.sim_cfg
    try:
        submission_dir, scores_dir = resolve_dirs(
            os.path.dirname(__file__), args.input_dir, args.output_dir
        )
    except Exception as e:
        msg = format_exception(e)
        result['error_msg'] = msg
        json.dump(result, open(scores_dir / "scores.json", 'w'), indent=2)
        raise AssertionError()

    # get agent and configuration of gym
    try:
        agent_spec, gym_cfg = load_agent_submission(submission_dir)
    except Exception as e:
        msg = format_exception(e)
        result['error_msg'] = msg
        json.dump(result, open(scores_dir / "scores.json", 'w'), indent=2)
        raise AssertionError()

    logger.info(f"Loaded user agent instance={agent_spec}")

    # simulation
    start_time = time.time()
    try:

        #---------------------------------------------------------------------------------
        # print('-'*100)
        # print(f'args {args}')
        logger.info('\n Trainer called here'+ '-'*100 + '\n')
        trainer = Trainer(args, agent_spec, simulator_cfg_file, gym_cfg, metric_period)
        trainer.train()
        logger.info(f"Total number of times called one_step_training {trainer.agent.num_steps_training}")
        scores = run_simulation(agent_spec, simulator_cfg_file, gym_cfg, metric_period, scores_dir, threshold)
    except Exception as e:
        msg = format_exception(e)
        result['error_msg'] = msg
        json.dump(result, open(scores_dir / "scores.json", 'w'), indent=2)
        raise AssertionError()

    # write result
    result['data']['total_served_vehicles'] = scores[0]
    result['data']['delay_index'] = scores[1]
    # result['data']['last_d_i'] = scores[2]
    result['success'] = True

    # cal time
    end_time = time.time()

    logger.info(f"total evaluation cost {end_time - start_time} s")

    # write score
    logger.info("\n\n")
    logger.info("*" * 40)

    json.dump(result, open(scores_dir / "scores.json", 'w'), indent=2)

    logger.info("Evaluation complete")


# trainer = Trainer()
# agent_specs is a dictionary, its key is "test" and its value is the agent instance