import os
import pickle
import random

import settings as s
import events as e

import numpy as np
import datetime as dt

from numba import jit
import copy

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def softmax(arr: np.ndarray) -> np.ndarray:
    sm_arr = np.copy(arr)
    denom = np.sum(np.exp(sm_arr))

    return np.exp(sm_arr)/denom


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
    if self.train or not os.path.isfile("MCA-V.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("MCA-V.pt", "rb") as file:
            self.V = pickle.load(file)
        with open("MCA-returns.pt", "rb") as file:
            self.returns = pickle.load(file)

        nz  = 0
        #self.logger.debug(str(self.V))
        for state in self.V:
            #self.logger.debug(self.V[state])
            if self.V[state] != 0:
                self.logger.debug(str(state) + " " + str(self.V[state]))
                nz += 1
        print("Found %d nonzero entries" % nz)
                
        

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .3

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.15, .15, .15, .15, .2, .2])

    

    options = []

    features = state_to_features(game_state)

    agent_pos = np.array(game_state["self"][3])
    self.logger.debug("AGENT_POS: " + str(agent_pos) + " STATE: " + str(features) + " VALUE: " +  str(self.V[tuple(features)]))
    for action in ACTIONS:
        if action == "UP":
            new_state = copy.deepcopy(game_state)
            if game_state["field"][tuple(agent_pos + [0,-1])] != 0:
                options += [-np.Inf]
            else:
                new_state["self"] = tuple(list(game_state["self"][:3]) +  [tuple([game_state["self"][3][0], game_state["self"][3][1]-1])])
                options += [self.V[tuple(state_to_features(new_state))]]
                self.logger.debug("UP " + str(new_state["self"][3])  + " " + str(state_to_features(new_state))  + " " + str(self.V[tuple(state_to_features(new_state))]))
        if action == "RIGHT":# and game_state["field"][game_state["self"][3][0] + 1, game_state["self"][3][1]] != -1:
            new_state = copy.deepcopy(game_state)
            if game_state["field"][tuple(agent_pos + [1,0])] != 0:
                options += [-np.Inf]
            else:
                new_state["self"] = tuple(list(game_state["self"][:3]) +  [tuple([game_state["self"][3][0] + 1, game_state["self"][3][1]])])
                options += [self.V[tuple(state_to_features(new_state))]]
                self.logger.debug("RIGHT " + str(new_state["self"][3]) + " " + str(state_to_features(new_state)) + " " + str(self.V[tuple(state_to_features(new_state))]))
        if action == "DOWN":# and game_state["field"][game_state["self"][3][0], game_state["self"][3][1] + 1] != -1:
            new_state = copy.deepcopy(game_state)
            if game_state["field"][tuple(agent_pos + [0,1])] != 0:
                options += [-np.Inf]
            else:
                new_state["self"] = tuple(list(game_state["self"][:3]) +  [tuple([game_state["self"][3][0], game_state["self"][3][1]+1])])
                options += [self.V[tuple(state_to_features(new_state))]]
                self.logger.debug("DOWN " + str(new_state["self"][3]) + " " + str(state_to_features(new_state)) + " " + str(self.V[tuple(state_to_features(new_state))]))
        if action == "LEFT": # and game_state["field"][game_state["self"][3][0] - 1, game_state["self"][3][1]] != -1:
            new_state = copy.deepcopy(game_state)
            if game_state["field"][tuple(agent_pos + [-1,0])] != 0:
                options += [-np.Inf]
            else:
                new_state["self"] = tuple(list(game_state["self"][:3]) +  [tuple([game_state["self"][3][0] - 1, game_state["self"][3][1]])])
                options += [self.V[tuple(state_to_features(new_state))]]
                self.logger.debug("LEFT " + str(new_state["self"][3]) + " " + str(state_to_features(new_state)) + " " + str(self.V[tuple(state_to_features(new_state))]))
        if action == "WAIT":
            new_state = copy.deepcopy(game_state)
            # slightly discourage waiting           
            val = self.V[tuple(state_to_features(new_state))]
            
            if val > 0:
                val *= 0.9
            else:
                val*=1.1
            options += [val]
            self.logger.debug("WAIT " + str(new_state["self"][3]) + " " + str(state_to_features(new_state)) + " " + str(self.V[tuple(state_to_features(new_state))]))
        if action == "BOMB":
            if not game_state["self"][2]:
                pass
                options += [-np.Inf]
            else:
                new_state = copy.deepcopy(game_state)
                new_state["bombs"].append((agent_pos, s.BOMB_TIMER - 1))
                new_features = state_to_features(new_state)
                options += [self.V[tuple(new_features)]]
                self.logger.debug("BOMB " + str(new_state["self"][3]) + " " + str(state_to_features(new_state)) + " " + str(self.V[tuple(state_to_features(new_state))]))

            
    self.logger.debug(str(options))
    options = np.array(options)
    
    options = softmax(options)

    # Comment or uncomment do (dis-)enable option to drop bombs
    #options[5] = 0
    #options = options/np.sum(options)

    self.logger.debug("OPTIONS: " + str(options))
    
    #return(ACTIONS[np.argmax(options)])
    action = np.random.choice(ACTIONS, p=options)
    self.logger.debug("ACTION: " + str(action))
    return action

def state_to_features(game_state: dict, events = None) -> np.array:
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

    # features: [distance to closest coin, in bomb range, timeout of closest bomb, distance to closest bomb, distance to enemy, distance to crate, in explosion, distance to closest safe spot, ]
    features = np.zeros([9])
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None    

    # Get position of agend
    agent_pos = game_state["self"][3]


    # find closest coin
    if len(game_state["coins"]) == 0:
        features[0] = -1
    else:
        closest_coin_pos = game_state["coins"][np.argmin(((game_state["coins"]-np.array(agent_pos))**2).sum(axis=1))]
        features[0] = np.min([5, np.sum(np.abs(np.array(agent_pos) - np.array(closest_coin_pos)))])

    #find closest bomb
    bomb_fields = []
    bomb_map = []
    if not game_state["bombs"] or len(game_state["bombs"]) == 0:
        features[1] = 0
        features[2] = -1
        features[3] = -1
    else:        
        bomb_map, timers = (np.array(game_state["bombs"])[:,0], np.array(game_state["bombs"])[:,1])
        bomb_fields = [(x[0] + i, x[1]) for x in np.array(game_state["bombs"])[:,0] for i in range(-s.BOMB_POWER,s.BOMB_POWER + 1)] \
                + [(x[0], x[1] + i) for x in np.array(game_state["bombs"])[:,0] for i in range(-s.BOMB_POWER, s.BOMB_POWER + 1)]
        d = np.array([[x[0],x[1]] for x in np.array(game_state["bombs"])[:,0]]) - np.array(agent_pos)
        features[3] = np.min([s.BOMB_POWER + 4,np.sum(np.abs(d[np.argmin(np.sum(d**2, axis=1))]))])
        features[1] = int(list(agent_pos) in bomb_fields)
        features[2] =  timers[np.argmin(np.sum(d**2,axis=1))]
    
    # Enemy distance
    if not game_state["others"] or len(game_state["others"]) == 0:
        features[4] = -1
    else:        
        others_fields = np.array([[x[3][0], x[3][1]] for x in game_state["others"]])
        d = others_fields - np.array(agent_pos)
        min_dist = np.sum(np.abs(d[np.argmin(np.sum(d**2,axis=1))]))
        features[4] = np.min([14, min_dist])
    
    #crates
    crates = []
    for i in range(1,16):
        for j in range(1,16):
            if game_state["field"][i,j] == 1:
                crates += [[i,j]]

    if len(crates) != 0:
        d = np.array(crates) - np.array(agent_pos)
        min_dist = np.sum(np.abs(d[np.argmin(np.sum(d**2,axis=1))]))
        #print("min dist:", min_dist)
        features[5] = np.min([5, min_dist])
    else:
        features[5] = 0

    explosions = []
    for i in range(1,16):
        for j in range(1,16):
            if game_state["explosion_map"][i,j] != 0:
                explosions += [[i,j]]

    if list(agent_pos) in explosions:
        features[6] = 1
    
    # Find (closest) walkable safe spots
    tested = []
    safe_spots = []
    paths = {}
    find_walkable_safe_spots(list(agent_pos), tested,safe_spots, bomb_fields, explosions, game_state["field"],paths)
    if len(safe_spots) == 0:
        features[7] = -1
        features[8] = 0
    else:
        d = [len(paths[key]) - 1 for key in paths if list(key) in safe_spots]
        features[7] = d[np.argmin(d)] if d[np.argmin(d)] < 6 else -1
        if features[7] <= features[2]:
            features[8] = 0
        else:
            features[8] = 1
    """if features[3] == 0:
        print("BOMBS:", bomb_fields, game_state["bombs"])
        print("EXPLOSIONS:", explosions)
        print("SAFE_SPOTS:", safe_spots)
        print(agent_pos, features)
        print()"""


    if events:
        if e.COIN_COLLECTED in events:
            features[0] = 0

    return features.astype(int)

def find_walkable_safe_spots(current_pos, tested, safe, bomb_fields, explosions, field, paths, path = []):
    path = copy.deepcopy(path)
    path += [current_pos]
    if tuple(current_pos) in paths.keys():
        paths[tuple(current_pos)] += path
    else:
        paths[tuple(current_pos)] = path

    if len(safe) > 5:
        return 0

    tested += [current_pos]
    if is_safe(current_pos,bomb_fields, explosions):
        safe += [current_pos] 
    if field[current_pos[0] - 1, current_pos[1]] == 0 and [current_pos[0] - 1, current_pos[1]] not in tested:

        find_walkable_safe_spots([current_pos[0]-1, current_pos[1]], tested,safe,bomb_fields,explosions,field,paths,path)
    if field[current_pos[0] + 1, current_pos[1]] == 0 and [current_pos[0] + 1, current_pos[1]] not in tested:
        find_walkable_safe_spots([current_pos[0]+1, current_pos[1]], tested,safe,bomb_fields,explosions,field,paths,path)
    if field[current_pos[0], current_pos[1] - 1] == 0 and [current_pos[0], current_pos[1] - 1] not in tested:
        find_walkable_safe_spots([current_pos[0], current_pos[1]-1], tested,safe,bomb_fields,explosions,field,paths,path)
    if field[current_pos[0], current_pos[1] + 1] == 0 and [current_pos[0], current_pos[1] + 1] not in tested:
        find_walkable_safe_spots([current_pos[0], current_pos[1]+1], tested,safe,bomb_fields,explosions,field,paths,path)

#def walkable_distance_to_safe_spot(agent_pos, safe_spots, )
    

def is_safe(current_pos,bomb_fields,explosions):
    if tuple(current_pos) not in bomb_fields and list(current_pos) not in explosions:
        return True
