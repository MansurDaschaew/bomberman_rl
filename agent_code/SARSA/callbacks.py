from _typeshed import IdentityFunction
import os
import pickle
import random

import numpy as np
import scipy.special


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

#do the setup/ get szaze 
def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)



# choose a eps greedy action
def eps_greedy(q_values, eps): 
    if np.random.random() < eps:
        return np.random.choice(len(q_values))
    else:
        return np.argmax(q_values)

#softmax 
def softmax(q_values, temp): 
    prob = scipy.special.softmax(q_values / temp)
    return np.random.choice(len(q_values, p=prob))

#choose a action according the desired policy
def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    action_map = game_state #maybe use a normalization for the game state?
    features = state_to_features(game_state)
    q_vals = np.dot(self.weights, features)

    # todo Exploration vs exploitation
    if self.train:
        if TRAIN_POLICY_TYPE == "EPS-GREEDY" :
            eps_train = np.interp(game_state["round"], eps_train_breaks, eps_train_vals)
            action_index = eps_greedy(q_vals, eps_train)
        elif TRAIN_POLICY_TYPE == "SOFTMAX":
            temp_train = np.interp(game_state["round"], inv_temp_train_breaks, 1/ np.array(inv_temp_train_vals))
            action_index = softmax(q_vals, temp_train)    
        else:
            raise NotImplementedError(f"choose valid policy type", {TRAIN_POLICY_TYPE})
    andom_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=self.model)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    pos = game_state["self"][3]
    field = game_state["field"]
    coins = game_state["coins"]
    bombs = game_state["bombs"]

    #GET INFO OF PLAYING FIELD
    # get the feature funcs from from features.py 
    
    #free tiles
    # free_tiles = get_free_tiles(field)
    
    
    #crate features
    #crate_features = get...
    
    #coin features

    #escape bomb and deadly koordinates
    
    # For example, you could construct several channels of equal shape, ...
    features = 0 #np.append(crate_features, is_next_to_crate, etc...)

    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    #return stacked_channels.reshape(-1)
    return features

print(eps_greedy([1,2,3], 0.05))

#train /setup 
if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    alpha = 0.1 #rate 
    gamma = 0.9 #discout factor
    eps = 1.0

    state = []
    for i in range(len(cartPosSpace)+1):
        for j in range(len(cartVelSpace)+1):
            for k in range(len(cartThetaSpace)+1):
                for l in range(len(cartThetaVelSpace)+1):
                    states.append((i,j,k,l))

    Q = {}
    for s in states: 
        for a in range(e):
            Q[(s,a)] = 0.0
    
    n = 16 #steps looking back (n step Sarsa)
    state_memory = np.zeros((n,4)) #states must be updated 
    action_memory = np.zeros(n)
    reward_memory = np.zeros(n)
    reward_memory = np.zeros(n)

    scores = []
    n_episodes = 400 #how many steps are beeing played
    for i in range(n_episodes): 
        done = False 
        score = 0.0
        t = 0 
        T = np.inf
        observation = choose_action(Q, observation, eps)
        action_memory[t%n] = action
        state_memory[t%n] = observation

        while not done:
            observation, reward, done, info = env.step(action)
            score += reward
            state_memory[(t+1)%n] = observation 
            reward_memory[(t+1)%n] = reward 
            if done: 
                T = t + 1
        
        action = choose_action(Q,observation,eps)
        action_memory[(t+1)%n] = action 
        tau = t - n - 1
        if tau >= 0:
            G = [gamma**(j-tau-1)*reward_memory[j%n] \
                for j in range(tau+1, min(tau+n,T)+1)]
            G = np.sum(G)
            if tau + n < T:
                s = get_state(state_memory[(tau+n%n)])
                a = int(action_memory[(tau+n)%n])
                G += gamma**n * Q([s,a])
            s = get_state(state_memory[tau%n])
            a = action_memory[tau%n]
            Q[(s,a)] += alpha * (G-Q[(s,a)])

    for tau in range(t-n+1,T):
        G = [gamma**(j-tau-1)*reward_memory[j%n]\
                for j in range(tau+1,min(tau+n, T)+1)]
        G = np.sum(G)
        if tau + n < T: 
            s = get_state(state_memory[(tau+n%n)])
            a = int(action_memory[(tau+n)%n])
            G += gamma**n * Q([s,a])
        
        s = get_state(state_memory[tau%n])
        a = action_memory[tau%n]
        Q[(s,a)] += alpha * (G-Q[(s,a)])

    scores.append(score)
    avg_score = np.mean(score[-1000:])
    epsilon = epsilon - 2 / n_episodes if epsilon > 0 else 0
    if i % 1000 == 0: 
        print('episode ', i, 'avg_score %.1f' % avg_score, 'epsilon %.2f' % epsilon)

    
    n = 16 #steps looking back (n step Sarsa)
    state_memory = np.zeros((n,4)) #states must be updated 
    action_memory = np.zeros(n)
    reward_memory = np.zeros(n)
    reward_memory = np.zeros(n)

    scores = []
    n_episodes = 400 #how many steps are beeing played
    for i in range(n_episodes): 
        done = False 
        score = 0.0
        t = 0 
        T = np.inf
        observation = choose_action(Q, observation, eps)
        action_memory[t%n] = action
        state_memory[t%n] = observation

        while not done:
            observation, reward, done, info = env.step(action)
            score += reward
            state_memory[(t+1)%n] = observation 
            reward_memory[(t+1)%n] = reward 
            if done: 
                T = t + 1
        
        action = choose_action(Q,observation,eps)
        action_memory[(t+1)%n] = action 
        tau = t - n - 1
        if tau >= 0:
            G = [gamma**(j-tau-1)*reward_memory[j%n] \
                for j in range(tau+1, min(tau+n,T)+1)]
            G = np.sum(G)
            if tau + n < T:
                s = get_state(state_memory[(tau+n%n)])
                a = int(action_memory[(tau+n)%n])
                G += gamma**n * Q([s,a])
            s = get_state(state_memory[tau%n])
            a = action_memory[tau%n]
            Q[(s,a)] += alpha * (G-Q[(s,a)])

    for tau in range(t-n+1,T):
        G = [gamma**(j-tau-1)*reward_memory[j%n]\
                for j in range(tau+1,min(tau+n, T)+1)]
        G = np.sum(G)
        if tau + n < T: 
            s = get_state(state_memory[(tau+n%n)])
            a = int(action_memory[(tau+n)%n])
            G += gamma**n * Q([s,a])
        
        s = get_state(state_memory[tau%n])
        a = action_memory[tau%n]
        Q[(s,a)] += alpha * (G-Q[(s,a)])

    scores.append(score)
    avg_score = np.mean(score[-1000:])
    epsilon = epsilon - 2 / n_episodes if epsilon > 0 else 0
    if i % 1000 == 0: 
        print('episode ', i, 'avg_score %.1f' % avg_score, 'epsilon %.2f' % epsilon)
