import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from agent import Agent

def ddpg_agent(n_episodes=2500, max_t=1000, env: UnityEnvironment = None):
    """Deep Deterministic Policy Gradient.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        env (UnityEnvironment): Unity Environment to solve
    """

    if not env: 
        env = UnityEnvironment('./Reacher_Linux/Reacher.x86_64')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    agent = Agent(state_size=state_size, action_size=action_size, random_seed=3)
    average_score = dict()             # average score list containing avg score every 100 episode
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    for i_episode in range(1, n_episodes+1):

        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations              # current state
        score = np.zeros(num_agents)

        for t in range(max_t):

            actions = agent.act(state)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_state = env_info.vector_observations         # get next state (for each agent)
            reward = env_info.rewards                         # get reward (for each agent)
            done = env_info.local_done                        # see if episode finished

            agent.step(state, actions, reward, next_state, done, t)
            score += env_info.rewards                         # update the score (for each agent)

            if np.any(done):
                break
        
        scores_window.append(np.mean(score))      # save most recent score
        scores.append(np.mean(score))             # save most recent score
        # print('\rEpisode {}\tAverage Score: {:.2f}, Memory : {}'.format(i_episode, np.mean(scores_window), len(agent.memory)), end="")
        if i_episode % 5 == 0:
            avg_score = np.mean(scores_window)
            average_score[i_episode] = avg_score
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            agent.save_models()
            break
    env.close()
    return scores
