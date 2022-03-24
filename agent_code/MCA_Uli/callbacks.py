import os
import pickle
import random

import numpy as np

import copy


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
            (self.V, self.returns) = pickle.load(file)
        
        self.V[(0,)] = 1


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    #print(options, np.argmax(options))

    #print(self.V[tuple(state_to_features(game_state))])
    if self.train: #and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    print(game_state["bombs"])
    

    options = []

    features = state_to_features(game_state)

    agent_pos = np.array(game_state["self"][3])
    #print("V", self.V)
    #print("pos:",agent_pos,game_state["field"][tuple(agent_pos)])
    for action in ACTIONS:
        new_state = copy.deepcopy(game_state)
        if action == "UP":#and game_state["field"][game_state["self"][3][0], game_state["self"][3][1] -1] != -1:
            if game_state["field"][tuple(agent_pos + [0,-1])] != 0:
                options += [-np.Inf]
            else:
                new_state["self"] = tuple(list(game_state["self"][:3]) +  [tuple([game_state["self"][3][0], game_state["self"][3][1]-1])])
                options += [self.V[tuple(state_to_features(new_state))]]
                self.logger.debug("UP " + str(new_state["self"][3])  + " " + str(state_to_features(new_state))  + " " + str(self.V[tuple(state_to_features(new_state))]))
        if action == "RIGHT":# and game_state["field"][game_state["self"][3][0] + 1, game_state["self"][3][1]] != -1:
            if game_state["field"][tuple(agent_pos + [1,0])] != 0:
                options += [-np.Inf]
            else:
                new_state["self"] = tuple(list(game_state["self"][:3]) +  [tuple([game_state["self"][3][0] + 1, game_state["self"][3][1]])])
                options += [self.V[tuple(state_to_features(new_state))]]
                self.logger.debug("RIGHT " + str(new_state["self"][3]) + " " + str(state_to_features(new_state)) + " " + str(self.V[tuple(state_to_features(new_state))]))
        if action == "DOWN":# and game_state["field"][game_state["self"][3][0], game_state["self"][3][1] + 1] != -1:
            if game_state["field"][tuple(agent_pos + [0,1])] != 0:
                options += [-np.Inf]
            else:
                new_state["self"] = tuple(list(game_state["self"][:3]) +  [tuple([game_state["self"][3][0], game_state["self"][3][1]+1])])
                options += [self.V[tuple(state_to_features(new_state))]]
                self.logger.debug("DOWN " + str(new_state["self"][3]) + " " + str(state_to_features(new_state)) + " " + str(self.V[tuple(state_to_features(new_state))]))
        if action == "LEFT": # and game_state["field"][game_state["self"][3][0] - 1, game_state["self"][3][1]] != -1:
            if game_state["field"][tuple(agent_pos + [-1,0])] != 0:
                options += [-np.Inf]
            else:
                new_state["self"] = tuple(list(game_state["self"][:3]) +  [tuple([game_state["self"][3][0] - 1, game_state["self"][3][1]])])
                options += [self.V[tuple(state_to_features(new_state))]]
                self.logger.debug("LEFT " + str(new_state["self"][3]) + " " + str(state_to_features(new_state)) + " " + str(self.V[tuple(state_to_features(new_state))]))
        if action == "WAIT":
            options += [self.V[tuple(state_to_features(new_state))]]
        #if action == "BOMB":
            
            
    options = np.array(options)
    self.logger.debug("1 " + str(options))
    #print("Between", ((options == 0) | (options == -np.Inf)).sum())
    if ((options == 0) | (options == -np.Inf)).sum() < 3:
        options[options != 0] = options[options != 0] - min(options[options != -np.Inf])
    #print(2,options)
    options = np.array(options) + 0.001
    #print(3,options)
    options = options/options[options != -np.Inf].sum()
    options[options == -np.Inf] = 0

    reduced_options = options
    #print(options, np.argmax(reduced_options))

    #self.logger.debug("Querying model for action.")
    options = np.concatenate([options])
    action = np.random.choice(ACTIONS, p=options)
    #print(action)
    return action


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

    # features: [distance to closest coin, distance to closest bomb, timeout of closest bomb]
    features = np.zeros([3])
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    

    # Find walkable directions -> left right up down
    agent_pos = game_state["self"][3]


    # find closest coin
    if len(game_state["coins"]) == 0:
        features[0] = -1
    else:
        closest_coin_pos = game_state["coins"][np.argmin(((game_state["coins"]-np.array(agent_pos))**2).sum(axis=1))]
        features[0] = np.sum(np.abs(np.array(agent_pos) - np.array(closest_coin_pos)))

    #find closest bomb
    if len(game_state["bombs"]) == 0:
        features[1] = -1
    else:
        bomb_map = np.array(game_state["bombs"])[:,0]
        print(np.array([[x[0],x[1]] for x in bomb_map]) - np.array(agent_pos))
        #bomb_map = np.array([np.array(game_state["bombs"][:][0]), game_state["bombs"][:][1]])
        #print(bomb_map)

        


    return features.astype(int)


