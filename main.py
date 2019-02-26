from unityagents import UnityEnvironment

import numpy as np
from collections import deque

import torch
import matplotlib.pyplot as plt

from agent import Agent


def ddpg_train(plot=False, env=None):

    if not env:
        env=UnityEnvironment(file_name="./Reacher_Linux/Reacher.x86_64")

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # environment information
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents in the environment
    n_agents = len(env_info.agents)
    print('Number of agents:', n_agents)
    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    # Agent 
    agent = Agent(state_size, n_agents, action_size, 4, './models/', loadModel=False)

    scores = []
    scores_window = deque(maxlen=100)
    n_episodes = 1000

    for episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]            # reset the environment
        states = env_info.vector_observations
        agent.reset()                                                # reset the agent noise
        score = np.zeros(n_agents)
        
        while True:
            actions = agent.act(states)
        
            env_info = env.step( actions )[brain_name]               # send the action to the environment                            
            next_states = env_info.vector_observations               # get the next state        
            rewards = env_info.rewards                               # get the reward        
            dones = env_info.local_done                              # see if episode has finished        

            agent.step(states, actions, rewards, next_states, dones)

            score += rewards                                         # update the score
        
            states = next_states                                     # roll over the state to next time step        
                                                        
            if np.any( dones ):                                        # exit loop if episode finished        
                break                                        

        if episode % 5 == 0:
            agent.save_model()

        scores.append(np.mean(score))
        scores_window.append(np.mean(score))

        print('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f}'.format(episode, np.mean(score), np.mean(scores_window)), end="")  
        
        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            agent.save_model()
            break

    if plot:  
        plt.plot(np.arange(1, len(scores)+1), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

    env.close()
    return scores


def play_solved(env=None):

    if not env:
        env=UnityEnvironment(file_name="./Reacher_Linux/Reacher.x86_64")

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # Number of agents in the environment
    n_agents = len(env_info.agents)
    print('Number of agents:', n_agents)
    # Number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    # Examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    # Agent 
    agent = Agent(state_size, n_agents, action_size, 4, './solved_models/', loadModel=True)

    for episode in range(3):
        env_info = env.reset(train_mode=False)[brain_name]        
        states = env_info.vector_observations       
        score = np.zeros(n_agents)               

        while True:
            actions = agent.act(states, add_noise=False)                    

            env_info = env.step(actions)[brain_name]        
            next_states = env_info.vector_observations     
            rewards = env_info.rewards       
            dones = env_info.local_done
            score += rewards
            states = next_states

            if np.any(dones):                              
                break

        print('Episode: \t{} \tScore: \t{:.2f}'.format(episode, np.mean(score)))      
    env.close()

if __name__ == "__main__":
    play_solved()
