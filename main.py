#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:06:47 2023

@author: oscar
"""
import os
import gym
import sys
import yaml
import torch
import warnings
import statistics
import numpy as np
from tqdm import tqdm 
from itertools import count
from collections import deque
import matplotlib.pyplot as plt
import time

sys.path.append('/home/kenzhi/SMARTS')
#from smarts.core.agent import AgentSpec
from smarts.zoo.agent_spec import AgentSpec
from smarts.env.hiway_env import HiWayEnv
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import NeighborhoodVehicles, RGB, OGM, DrivableAreaGridMap

from smarts import sstudio
import pathlib

from DQNcopy import DQN
from SAC import SAC
from Tnetwork import device

def plot_animation_figure():
    plt.figure()
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_list)
    print("reward_list", reward_list)
    print("reward_mean_list", reward_mean_list)
    plt.plot(reward_mean_list)
    plt.pause(10)
    plt.tight_layout()
    plt.show()
    plt.savefig("outputRASAC.jpg")

def preprocess(s, a, r, s_):
    state = s
    next_state  = s_
    action = a.cpu().numpy().squeeze().astype(np.int32)
    reward = np.float32(r)

    
    return state, action, reward, next_state

def evaluate(network, eval_episodes=10, epoch=0):
    ep = 0
    success = int(0)
    avg_reward_list = []
    cumulate_flag = True
    while ep < eval_episodes:
        obs = env.reset()
        s = observation_adapter(obs[AGENT_ID])
        done = False
        reward_total = 0.0 
        frame_skip = 3

        for t in count():
            
            if t > MAX_NUM_STEPS:
                print('Max Steps Done.')
                break
       
            if t < frame_skip:
                ##### Select and perform an action #####
                _, a = agent.select_action_deterministic(s)
                action = {AGENT_ID:action_adapter(a)}
                next_state, reward, done, info = env.step(action)
                s_ = observation_adapter(next_state[AGENT_ID])
                done = done[AGENT_ID]
                r = reward_adapter(next_state[AGENT_ID], a, done)
                s = s_

            ##### Select and perform an action ######
            _, a = agent.select_action_deterministic(s)
       
            action = {AGENT_ID:action_adapter(a)}
            next_state, reward, done, info = env.step(action)
            s_ = observation_adapter(next_state[AGENT_ID])
            done = done[AGENT_ID]
            r = reward_adapter(next_state[AGENT_ID], a, done)
       
            if done and not info[AGENT_ID]['env_obs'].events.reached_goal:
                r -= 1.0
       
            lane_name = info[AGENT_ID]['env_obs'].ego_vehicle_state.lane_id
            lane_id = info[AGENT_ID]['env_obs'].ego_vehicle_state.lane_index
       
            ##### Preprocessing ######
            state, action, reward, next_state = preprocess(s, a, r, s_)
       
            reward_total += reward                    
            s = s_
       
            if done:
                if info[AGENT_ID]['env_obs'].events.reached_goal:
                #if bool(len(info[AGENT_ID]['env_obs'].events.collisions)) == False:    
                    success += 1
                
                print('\n|=== EVALUATION ===',
                      '\n|Epoc:', ep,
                      '\n|Step:', t,
                      '\n|Collision:', bool(len(info[AGENT_ID]['env_obs'].events.collisions)),
                      '\n|Off Road:', info[AGENT_ID]['env_obs'].events.off_road,
                      '\n|Goal:', info[AGENT_ID]['env_obs'].events.reached_goal,
                      #'\n|Goal:', bool(not(len(info[AGENT_ID]['env_obs'].events.collisions))),
                      '\n|Off Route:', info[AGENT_ID]['env_obs'].events.off_route,
                      '\n|Reward Total:', reward_total,
                      '\n|Algo:', name,
                      '\n|seed:', seed,
                      '\n|Env:', env_name)
                break
            
        ep += 1
        avg_reward_list.append(reward_total)
        print("\n..............................................")
        print("%i Loop, Steps: %i, Avg Reward (Reward Total): %f, Success No. : %i " % (ep, t, reward_total, success))
        print("..............................................")

    reward = statistics.mean(avg_reward_list)
    print("\n..............................................")
    print("Average Reward over %i Evaluation Episodes, At Epoch: %i, Avg Reward:%f, Success No.: %i" % (eval_episodes, ep, reward, success))
    print("..............................................")
    return reward

# observation space
def observation_adapter(env_obs):
    global states
    #print(env_obs, "env_obs")
    np.set_printoptions(threshold=sys.maxsize)
    new_obs = env_obs.top_down_rgb[1]   #/255.0
    #
    #print("new_obs", len(new_obs))
    
    # for i in range(len(new_obs)):
    #     for j in range(len(new_obs[i])):
    #         for k in range(len(new_obs[i][j])):
    #             print("k", k)
    #             print(new_obs[i][j])
    #             if new_obs[i][j][k] != 0.0:
    #                 print("obseravtion is not 0 0 0")
    #                 plt.imshow(new_obs)
    #                 time.sleep(2)
    #                 plt.close()
    
    #new_obs = env_obs.occupancy_grid_map[1]
    #print(new_obs, "new_obs")
    #plt.imshow(new_obs)
    #time.sleep(5)
    #plt.close()
    states[:, :, 0:3] = states[:, :, 3:6]
    states[:, :, 3:6] = states[:, :, 6:9]
    states[:, :, 6:9] = new_obs
    ogm = env_obs.occupancy_grid_map[1] 
    drivable_area = env_obs.drivable_area_grid_map[1]
    
    if env_obs.events.collisions or env_obs.events.reached_goal:
        states = np.zeros(shape=(screen_size, screen_size, 9))
    return np.array(states, dtype=np.uint8)


# reward function
def reward_adapter(env_obs, action, done, engage=False):
    ego_obs = env_obs.ego_vehicle_state
    #ego_lat_error = ego_obs.lane_position.t
    ego_speed = env_obs.ego_vehicle_state.speed
    lane_name = ego_obs.lane_id
    lane_id = ego_obs.lane_index

    '''
    if env_name == 'Merge':
        if lane_name == 'gneE6_0' and action > 1:
        # if lane_name == 'gneE6_0' and action == 2:
            off_road = - 1.0
            print('Off lane at junction')
        elif (lane_name == 'gneJ5_3_0' or lane_name == 'gneE4_0') and action == 2:
            off_road = - 1.0
            print('off lane at link, right turn!')
        elif lane_name == 'gneE4_2' and action == 3:
            off_road = - 1.0
            print('off lane at highway, left turn!')
        else:
            off_road = 0.0

        if lane_name == 'gneE4_2':
            target_lane = 0.2
        elif lane_name == 'gneE4_0':
            target_lane = - 0.4
        elif lane_name == 'gneE4_1':
            target_lane = - 0.4
        else:
            target_lane = 0.0
        
        heuristic = ego_speed * 0.002

    elif env_name =='leftturn':
        if lane_name == 'E0_0' and action > 1:
            off_road = - 1.0
            print('Off lane at E0_0')
        elif lane_name == 'E1_0' and ego_lat_error < 0.0 and action == 2:
            off_road = - 1.0
            print('Off lane at E1_0')
        elif lane_name == 'E1_1' and ego_lat_error > 0.0 and action == 3:
            off_road = - 1.0
            print('Off lane at E1_1')
        else:
            off_road = 0.0
        
        heuristic = ego_speed * 0.002 if ego_speed > 2.0 else - 0.05'''
        
    #else:
    off_road = 0.0
    target_lane = 0.0
    heuristic = env_obs.ego_vehicle_state.speed * 0.002
    
    if done:
        if env_obs.events.reached_goal:
            print('\n Goal')
            goal = 2.0
        else:
            goal = -2.0
    else:
        goal = 0.0
    '''
    if (done) and (not(env_obs.events.collisions)):
        goal = 4.0 #2.0
    else:
        goal =0'''

    if env_obs.events.collisions:
        print('\n crashed')
        crash = -2
    else:
        crash = 0.0
        
    if env_obs.events.on_shoulder:
        print('\n on_shoulder')
        performance = - 0.04
    else:
        performance = 0.0
        
    #if action > 0:
    if action > 0:
        penalty = -0.05
    else:
        penalty = 0.0

    time_taken = -0.05

    return heuristic + off_road + goal + crash + performance + penalty + time_taken
    #return heuristic + off_road + goal + crash + performance + penalty + time_taken

# action space
def action_adapter(model_action): 

    # discretization
    if model_action == 0:
        lane = 'keep_lane'
    elif model_action == 1:
        lane = 'slow_down'
    elif model_action == 2:
        lane = 'change_lane_right'
    else:
        lane = 'change_lane_left'

    return lane

# information
def info_adapter(observation, reward, info):
    return info

def interaction(COUNTER):
    save_threshold = 0.0
    trigger_reward = 10 #10
    trigger_epoc = 200 #200
    goal_counter = 0

    for epoc in tqdm(range(1, MAX_NUM_EPOC+1), ascii=True):
        reward_total = 0.0 
        obs = env.reset()
        s = observation_adapter(obs[AGENT_ID])
        frame_skip = 3
        
        for t in count():
            
            if t > MAX_NUM_STEPS:
                print('Max Steps Done.')
                break
            
            ##### Skip first several frames #####
            if t <= frame_skip:
                ###### Select and perform an action #######
                distribution, a = agent.select_action(s, epoc)
                action = {AGENT_ID:action_adapter(a)}
                engage = int(0)
                next_state, reward, done, info = env.step(action)
                #print("next_state", next_state)
                s_ = observation_adapter(next_state[AGENT_ID])
                done = done[AGENT_ID]
                r = reward_adapter(next_state[AGENT_ID], a, done, engage)
                s = s_

            ##### Select and perform an action ######
            distribution, rl_a = agent.select_action(s, epoc)
            
            ###### Assign final action ######
            a = rl_a
            engage = int(0)
            
            ##### Interaction #####
            action = {AGENT_ID:action_adapter(a)}
            next_state, reward, done, info = env.step(action)
            
            #plt.imshow(next_state[AGENT_ID].top_down_rgb[1])
            #plt.show()
            # time.sleep(0.2)
            
            s_ = observation_adapter(next_state[AGENT_ID])
            done = done[AGENT_ID]
            r = reward_adapter(next_state[AGENT_ID], a, done, engage)

            ##### Preprocessing ######
            state, action, reward, next_state = preprocess(s, a, r, s_)
            ##### Store the transition in memory ######
            agent.store_transition(state, action, reward, next_state, engage, done)
            reward_total += reward
    
            if epoc >= THRESHOLD:
                if t % Q_FREQ == 0:
                    agent.optimize_network()

    
            if epoc >= THRESHOLD and done:
                if epoc % 2 == 0:
                    train_durations.append(epoc)

                goal_list.append(goal_counter/(epoc))                

                #agent.scheduler.step()
                    
            s = s_
    
            if done:
                
                if info[AGENT_ID]['env_obs'].events.reached_goal:
                    goal_counter += 1
                    print("Reached Goal! Goal_Counter:", goal_counter, " Epoc:", epoc)
                '''if bool(len(info[AGENT_ID]['env_obs'].events.collisions)) == False:
                    goal_counter += 1
                    print("Reached Goal! Goal_Counter:", goal_counter, " Epoc:", epoc)'''

                reward_list.append(reward_total)
                reward_mean_list.append(np.mean(reward_list[-20:]))
                
                ###### Evaluating the performance of current model ######
                if reward_mean_list[-1] >= trigger_reward and epoc > trigger_epoc:
                    trigger_reward = reward_mean_list[-1]
                    print("Evaluating the Performance.")
                    avg_reward = evaluate(agent, EVALUATION_EPOC)
                    if avg_reward > save_threshold:
                        print('Save the model at %i epoch, reward is: %f' % (epoc, avg_reward))
                        saved_epoc = epoc
                        torch.save(agent.policy_net.state_dict(), os.path.join('trained_network/'+ scenario,
                                  name+'_memo'+str(MEMORY_CAPACITY)+'_epoc'+
                                  str(MAX_NUM_EPOC)+'_seed'
                                  + str(seed)+'_' + str(agent.lr_p)+'_'+env_name+'_policynet.pkl'))
                        save_threshold = avg_reward

                print('\n|',
                      '\n|Epoc:', epoc,
                      '\n|Step:', t,
                      '\n|Goal Rate:', goal_list[-1],
                      '\n|Goal:', info[AGENT_ID]['env_obs'].events.reached_goal,
                      #'\n|Goal:', bool(not(len(info[AGENT_ID]['env_obs'].events.collisions))),
                      '\n|Collision:', bool(len(info[AGENT_ID]['env_obs'].events.collisions)),
                      '\n|Off Road:', info[AGENT_ID]['env_obs'].events.off_road,
                      '\n|Off Route:', info[AGENT_ID]['env_obs'].events.off_route,
                      '\n|ExpR:', agent.eps_threshold,
                      #'\n|Temperature:', agent.temperature_copy,
                      '\n|R:', reward_total,
                      '\n|Algo:', name,
                      '\n|seed:', seed,
                      '\n|Env:', env_name)
    
                s = env.reset()
                reward_total = 0
                break
        
        ''''
        if epoc == 30:
            plot_animation_figure()
    
        elif (epoc % SAVE_INTERVAL == 0):
            np.save(os.path.join('store/' + scenario, 'reward_memo'+str(MEMORY_CAPACITY)+
                                  '_epoc'+str(MAX_NUM_EPOC)+'_seed'+ str(seed) +
                                  '_' + str(agent.lr_p)+'_'+env_name+'_' + name),
                    [reward_mean_list], allow_pickle=True, fix_imports=True)
    
            np.save(os.path.join('store/' + scenario, 'steps_memo'+str(MEMORY_CAPACITY)+
                                  '_epoc'+str(MAX_NUM_EPOC)+'_seed'+ str(seed) +
                                  '_' + str(agent.lr_p)+'_'+env_name+'_' + name),
                    [train_durations], allow_pickle=True, fix_imports=True) 
                    '''     
                    
    print('Complete')
    return save_threshold

if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    plt.ion()

    path = os.getcwd()
    yaml_path = os.path.join(path, 'configcopy.yaml')
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    ##### Individual parameters for each model ######f
    mode = 'SAC'
    mode_param = config[mode]
    name = mode_param['name']

    ###### Default parameters for DRL ######
    IMPORTANTSAMPLING  = config['IMPORTANTSAMPLING']
    ENTROPY = config['ENTROPY']
    THRESHOLD = config['THRESHOLD']
    TARGET_UPDATE = config['TARGET_UPDATE']
    BATCH_SIZE = config['BATCH_SIZE']
    GAMMA = config['GAMMA']
    MEMORY_CAPACITY = config['MEMORY_CAPACITY']
    PREFERENCE_FREQ = config['PREFERENCE_FREQ']
    Q_FREQ = config['Q_FREQ']
    EPS_START = config['EPS_START']
    EPS_END = config['EPS_END']
    EPS_DECAY= config['EPS_DECAY']
    MAX_NUM_EPOC = config['MAX_NUM_EPOC']
    MAX_NUM_STEPS = config['MAX_NUM_STEPS']
    PLOT_INTERVAL = config['PLOT_INTERVAL']
    SAVE_INTERVAL = config['SAVE_INTERVAL']
    DECISION_VARIABLE = config['DECISION_VARIABLE']
    EVALUATION_EPOC = config['EVALUATION_EPOC']

    #### Environment specs ####
    env_name = config['env_name']

    if env_name == 'LeftTurn':
        scenario = 'LeftTurn'
    elif env_name == 'Test':
        scenario = 'Test'
    elif env_name == 'loop':
        scenario = 'loop'
    elif env_name == 'Roundabout':
        scenario = 'Roundabout'
    elif env_name == 'Intersection':
        scenario = 'Intersection'
    else:
        scenario = 'RampMerge'

    if not os.path.exists("./store/" + scenario):
        os.makedirs("./store/" + scenario)

    if not os.path.exists("./trained_network/" + scenario):
        os.makedirs("./trained_network/" + scenario)
    '''
    scenarios = [
        str(
            pathlib.Path(__file__).absolute().parents[1]
            / "Scenario"
            / "LeftTurn"
            )
        ]

    sstudio.build_scenario(scenario=scenarios)'''

    screen_size = config['screen_size']
    view = config['view']
    AGENT_ID = config['AGENT_ID']
    ACTION_SPACE = gym.spaces.Discrete(DECISION_VARIABLE)
    OBSERVATION_SPACE = gym.spaces.Box(low=0, high=1, shape=(screen_size, screen_size, 9))
    states = np.zeros(shape=(screen_size, screen_size, 9))

    ##### Define agent interface #######
    agent_interface = AgentInterface(
        max_episode_steps=MAX_NUM_STEPS,
        road_waypoints=True,
        neighborhood_vehicles=NeighborhoodVehicles(radius=100),
        rgb=RGB(screen_size, screen_size, view/screen_size),
        ogm=OGM(screen_size, screen_size, view/screen_size),
        drivable_area_grid_map=DrivableAreaGridMap(screen_size, screen_size, view/screen_size),
        action=ActionSpaceType.Lane,
    )
    ###### Define agent specs ######
    agent_spec = AgentSpec(
        interface=agent_interface
    )
    legend_bar = []
    seed_list = [0,97,98,99]
    ##### Train #####
    #for i in range(0, 3):
    #for i in range (0,1):
    seed = seed_list[0]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ##### Create Env ######
    scenario_path = ['Scenario/' + str(scenario)]
    #env = HiWayEnv(scenarios=scenario_path, agent_specs={AGENT_ID: agent_spec},
    #            headless=False, visdom=False, sumo_headless=True, seed=seed)
    env = HiWayEnv(scenarios=scenario_path, agent_specs={AGENT_ID: agent_spec},
                headless=False, visdom=False, sumo_headless=False,)
    env.observation_space = OBSERVATION_SPACE
    env.action_space = ACTION_SPACE
    env.agent_id = AGENT_ID
    obs = env.reset()
    img_h, img_w, channel = screen_size, screen_size, 9
    n_obs = img_h * img_w * channel
    n_actions = env.action_space.n

    # create RL agent"
    
    if name == "dqn":
        agent = DQN(img_h, img_w, channel, n_obs, n_actions,
                    IMPORTANTSAMPLING, ENTROPY, BATCH_SIZE,
                    GAMMA, EPS_START, EPS_END, EPS_DECAY, THRESHOLD, MEMORY_CAPACITY, seed)
    elif name == "sac":
        agent = SAC(img_h, img_w, channel, n_obs, n_actions,
                    IMPORTANTSAMPLING, ENTROPY, BATCH_SIZE,
                    GAMMA, EPS_START, EPS_END, EPS_DECAY, THRESHOLD, MEMORY_CAPACITY, seed)
    else:
        print("Agent ",name," is not detected.")


    legend_bar.append('seed'+str(seed))

    train_durations = []
    train_durations_mean_list = []
    reward_list = []
    reward_mean_list = []
    goal_list = []
    goal_list.append(0.0)

    print('\nThe object is:', mode, '\n|Seed:', agent.seed)
            
    success_count = 0

    save_threshold = interaction(success_count)

    np.save(os.path.join('store/' + scenario, 'reward_memo'+str(MEMORY_CAPACITY)+
                            '_epoc'+str(MAX_NUM_EPOC)+'_seed'+ str(seed) +
                            '_' + str(agent.lr_p)+'_'+env_name+'_' + name),
            [reward_mean_list], allow_pickle=True, fix_imports=True)

    np.save(os.path.join('store/' + scenario, 'steps_memo'+str(MEMORY_CAPACITY)+
                            '_epoc'+str(MAX_NUM_EPOC)+'_seed'+ str(seed) +
                            '_' + str(agent.lr_p)+'_'+env_name+'_' + name),
            [train_durations], allow_pickle=True, fix_imports=True)           

    print("Evaluating the Performance.")
    avg_reward = evaluate(agent, EVALUATION_EPOC)
    if avg_reward > save_threshold:
        print('Save the model!')
        torch.save(agent.policy_net.state_dict(), os.path.join('trained_network/' + scenario,
                name+'_memo'+str(MEMORY_CAPACITY)+'_epoc'+
                str(MAX_NUM_EPOC)+'_seed'
                + str(seed)+'_' + str(agent.lr_p)+'_'+env_name+'_policynet.pkl'))

    plot_animation_figure()
    env.close()



