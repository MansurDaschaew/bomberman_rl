import os
import pickle
import random

import numpy as np

import copy


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']


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
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2])
    

    options = []

    features = state_to_features(game_state)
    print(features)
    print(game_state["self"][3])
    for action in ACTIONS:
        new_state = copy.deepcopy(game_state)
        if action == "UP":#and game_state["field"][game_state["self"][3][0], game_state["self"][3][1] -1] != -1:
            if not features[2]:
                options += [-np.Inf]
            else:
                new_state["self"] = tuple(list(game_state["self"][:3]) +  [tuple([game_state["self"][3][0], game_state["self"][3][1]-1])])
                options += [self.V[tuple(state_to_features(new_state))]]
                print("UP", new_state["self"][3], state_to_features(new_state),self.V[tuple(state_to_features(new_state))])
        if action == "RIGHT":# and game_state["field"][game_state["self"][3][0] + 1, game_state["self"][3][1]] != -1:
            if not features[1]:
                options += [-np.Inf]
            else:
                new_state["self"] = tuple(list(game_state["self"][:3]) +  [tuple([game_state["self"][3][0] + 1, game_state["self"][3][1]])])
                options += [self.V[tuple(state_to_features(new_state))]]
                print("RIGHT", new_state["self"][3], state_to_features(new_state),self.V[tuple(state_to_features(new_state))])
        if action == "DOWN":# and game_state["field"][game_state["self"][3][0], game_state["self"][3][1] + 1] != -1:
            if not features[3]:
                options += [-np.Inf]
            else:
                new_state["self"] = tuple(list(game_state["self"][:3]) +  [tuple([game_state["self"][3][0], game_state["self"][3][1]+1])])
                options += [self.V[tuple(state_to_features(new_state))]]
                print("DOWN", new_state["self"][3], state_to_features(new_state),self.V[tuple(state_to_features(new_state))])
        if action == "LEFT": # and game_state["field"][game_state["self"][3][0] - 1, game_state["self"][3][1]] != -1:
            if not features[0]:
                options += [-np.Inf]
            else:
                new_state["self"] = tuple(list(game_state["self"][:3]) +  [tuple([game_state["self"][3][0] - 1, game_state["self"][3][1]])])
                options += [self.V[tuple(state_to_features(new_state))]]
                print("LEFT", new_state["self"][3], state_to_features(new_state),self.V[tuple(state_to_features(new_state))])
        #if action == "WAIT":
            #options += [self.V[tuple(state_to_features(new_state))]]
    options = np.array(options)
    #print(options)
    
    #options = options/options[options != np.Inf].sum()
    #print(options)
    #print(options)
    #options[options == -np.Inf] = 0
    #print(options)

    allowed_directions = features[:4][[2,1,3,0]]
    #reduced_options = options[allowed_directions.astype(bool) & (options != 0)]
    #reduced_options = options[(options != 0)]
    reduced_options = options
    #print(options[options != 0])
    #print(state_to_features(game_state))
    print(options, np.argmax(reduced_options))
    print(ACTIONS[list(options).index(reduced_options[np.argmax(reduced_options)])])
    return ACTIONS[list(options).index(reduced_options[np.argmax(reduced_options)])]

    #self.logger.debug("Querying model for action.")
    options = np.concatenate([options, [0]])
    return np.random.choice(ACTIONS, p=options)


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
    features = np.zeros([9])
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    

    # Find walkable directions -> left right up down
    agent_pos = game_state["self"][3]

    if agent_pos == (15,16) or agent_pos == (16,15):
        agent_pos = (15,15)

    # As Wall is marked as -1 and Path 1, respectively, we cann add one (for now)
    # To find out if we can walk in a direction or not
    #if game_state["step"] == 1 or game_state["step"] == 2:
        #print(agent_pos, game_state["field"][agent_pos])
    features[0] = game_state["field"][agent_pos[0]-1, agent_pos[1]] + 1
    features[1] = game_state["field"][agent_pos[0]+1, agent_pos[1]] + 1
    features[2] = game_state["field"][agent_pos[0], agent_pos[1] - 1] + 1
    features[3] = game_state["field"][agent_pos[0], agent_pos[1] + 1] + 1
    

    # Another Feature: find direction of closest coin (up down left right)
    # Square Sum of position difference and find minimal distance coin
    # TODO (optionally): Allow multiple coins that have the same
    #      distance and then allow multiple directions

    # find closest coin
    closest_coin_pos = game_state["coins"][np.argmin(((game_state["coins"]-np.array(agent_pos))**2).sum(axis=1))]

    # check if closest coin is in x or y direction
    x_or_y = np.argmax(np.abs(np.array(agent_pos) - np.array(closest_coin_pos)))

    # depending on X or Y directions, check if left/right or up/down
    if x_or_y:
        if (np.array(closest_coin_pos) - np.array(agent_pos))[x_or_y] < 0:
            features[4+2] = 1
        elif (np.array(closest_coin_pos) - np.array(agent_pos))[x_or_y] > 0:
            features[4+3] = 1
    else:
        if (np.array(closest_coin_pos) - np.array(agent_pos))[x_or_y] < 0:
            features[4+0] = 1
        elif (np.array(closest_coin_pos) - np.array(agent_pos))[x_or_y] > 0:
            features[4+1] = 1
    features[8] = np.sum(np.abs(np.array(agent_pos) - np.array(closest_coin_pos)))
    #test comment

    return features.astype(int)



