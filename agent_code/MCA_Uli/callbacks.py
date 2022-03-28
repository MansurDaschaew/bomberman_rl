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
    Initial setup loading data from model, if training not set

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    
    # check if either training or file exists
    if self.train or not os.path.isfile("MCA-V.pt"):
        self.logger.info("Setting up model from scratch.")
    else:
        self.logger.info("Loading model from saved state.")
        with open("MCA-V.pt", "rb") as file:
            self.V = pickle.load(file)
        with open("MCA-returns.pt", "rb") as file:
            self.returns = pickle.load(file)
        

        # debug info on how many non zero values there are.
        # run only once. Only helpflul for debug
        nz  = 0
        for state in self.V:
            if self.V[state] != 0:
                self.logger.debug(str(state) + " " + str(self.V[state]))
                nz += 1
        self.logger.debug("Found %d nonzero entries" % nz)
                
        

def act(self, game_state: dict) -> str:
    """
    Makes a decision from a given state, which option in ACTIONS is best

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    # 20% of steps are random during training
    random_prob = .2

    # explore possibilities if in training mode (20% probability)
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # walk in any direction 60% (divided into each direction), wait 5% (as it is boring)
        # drop bombs 35% -> more interesting case to learn from
        return np.random.choice(ACTIONS, p=[.15, .15, .15, .15, .05, .35])    

    # Init array for available options
    options = []

    # get current feature state
    features = state_to_features(game_state)

    # get position of agent
    agent_pos = np.array(game_state["self"][3])
    
    self.logger.debug("AGENT_POS: " + str(agent_pos) + " STATE: " + str(features) + " VALUE: " +  str(self.V[tuple(features)]))

    # Check State-Values for all possible actions
    #
    # Disallowing illegal moves is hardcoded. Could have been learned as well
    # but as we needed to reduce the feature space, hardcoding not to take
    # illegal moves seemed to be a good choice
    for action in ACTIONS:
        if action == "UP":
            new_state = copy.deepcopy(game_state) # deepcopy state such that modifications don't propagate to other actions
            if game_state["field"][tuple(agent_pos + [0,-1])] != 0:
                options += [-np.Inf] # assing value of - inf to force softmax to not take this option, if it's an illegal move
            else:
                # modify new state according to action
                new_state["self"] = tuple(list(game_state["self"][:3]) +  [tuple([game_state["self"][3][0], game_state["self"][3][1]-1])])
                # add corresponding value to options
                options += [self.V[tuple(state_to_features(new_state))]]
                self.logger.debug("UP " + str(new_state["self"][3])  + " " + str(state_to_features(new_state))  + " " + str(self.V[tuple(state_to_features(new_state))]))
        if action == "RIGHT":# and game_state["field"][game_state["self"][3][0] + 1, game_state["self"][3][1]] != -1:
            new_state = copy.deepcopy(game_state)
            if game_state["field"][tuple(agent_pos + [1,0])] != 0:
                options += [-np.Inf]
            else:
                # modify new state according to action
                new_state["self"] = tuple(list(game_state["self"][:3]) +  [tuple([game_state["self"][3][0] + 1, game_state["self"][3][1]])])
                # add corresponding value to options
                options += [self.V[tuple(state_to_features(new_state))]]
                self.logger.debug("RIGHT " + str(new_state["self"][3]) + " " + str(state_to_features(new_state)) + " " + str(self.V[tuple(state_to_features(new_state))]))
        if action == "DOWN":# and game_state["field"][game_state["self"][3][0], game_state["self"][3][1] + 1] != -1:
            new_state = copy.deepcopy(game_state)
            if game_state["field"][tuple(agent_pos + [0,1])] != 0:
                options += [-np.Inf]
            else:
                # modify new state according to action
                new_state["self"] = tuple(list(game_state["self"][:3]) +  [tuple([game_state["self"][3][0], game_state["self"][3][1]+1])])
                options += [self.V[tuple(state_to_features(new_state))]]
                self.logger.debug("DOWN " + str(new_state["self"][3]) + " " + str(state_to_features(new_state)) + " " + str(self.V[tuple(state_to_features(new_state))]))
        if action == "LEFT": # and game_state["field"][game_state["self"][3][0] - 1, game_state["self"][3][1]] != -1:
            new_state = copy.deepcopy(game_state)
            if game_state["field"][tuple(agent_pos + [-1,0])] != 0:
                options += [-np.Inf]
            else:
                # modify new state according to action
                new_state["self"] = tuple(list(game_state["self"][:3]) +  [tuple([game_state["self"][3][0] - 1, game_state["self"][3][1]])])
                # add corresponding value to options
                options += [self.V[tuple(state_to_features(new_state))]]
                self.logger.debug("LEFT " + str(new_state["self"][3]) + " " + str(state_to_features(new_state)) + " " + str(self.V[tuple(state_to_features(new_state))]))
        if action == "WAIT":
            new_state = copy.deepcopy(game_state)
            val = self.V[tuple(state_to_features(new_state))]
         
            # slightly discourage waiting           
            if val > 0:
                val *= 0.8
            else:
                val*=1.2
            # add corresponding value to options
            options += [val]
            self.logger.debug("WAIT " + str(new_state["self"][3]) + " " + str(state_to_features(new_state)) + " " + str(self.V[tuple(state_to_features(new_state))]))
        if action == "BOMB":
            if not game_state["self"][2]: # Again disallow illegal action
                pass
                options += [-np.Inf]
            else:
                new_state = copy.deepcopy(game_state)
                # modify new state according to action
                new_state["bombs"].append((agent_pos, s.BOMB_TIMER - 1))
                new_features = state_to_features(new_state)
                # add corresponding value to options
                options += [self.V[tuple(new_features)]]
                self.logger.debug("BOMB " + str(new_state["self"][3]) + " " + str(state_to_features(new_state)) + " " + str(self.V[tuple(state_to_features(new_state))]))

            
    options = np.array(options)
   
    # Use softmax to map values of all options to a probability space
    options = softmax(options)

    # DEBUG OPTION: used when playing around to disable dropping bombs avoiding suicide
    # Comment or uncomment do (dis-)enable option to drop bombs
    #options[5] = 0
    #options = options/np.sum(options)

    self.logger.debug("OPTIONS: " + str(options))
    
    # take Choice of actions
    action = np.random.choice(ACTIONS, p=options)

    self.logger.debug("ACTION: " + str(action))
    return action

def state_to_features(game_state: dict, events = None) -> np.array:
    """
    Turns a given state into a set of features.
    As with the Monte-Carlo RL we desperately need to reduce the feature space,
    we make high use of this function.
    
    :param game_state:  A dictionary describing the current game board.
    :param events: List of events that happend (required for learning to collect coins)
    :return: np.array
    """

    # features: [distance to closest coin, in bomb range, timeout of closest bomb, distance to closest bomb, 
    #            distance to enemy, distance to crate, in explosion, distance to closest safe spot, safespot reachable within timer]
    features = np.zeros([9])

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None    

    # Get position of agend
    agent_pos = game_state["self"][3]


    # FEATURE[0]: closest coin distance
    if len(game_state["coins"]) == 0:
        features[0] = -1
    else:
        closest_coin_pos = game_state["coins"][np.argmin(((game_state["coins"]-np.array(agent_pos))**2).sum(axis=1))]
        features[0] = np.min([5, np.sum(np.abs(np.array(agent_pos) - np.array(closest_coin_pos)))])

    # FEATURE[1] field in BOMB_RANGE (when exploding at some point)
    # FEATURE[2] Timeout of closest bomb
    # FEATURE[3] Distance to closest bomb
    # find closest bomb and assign values to features
    bomb_fields = []
    bomb_map = []

    # assign values if no bomb is present
    if not game_state["bombs"] or len(game_state["bombs"]) == 0:
        features[1] = 0
        features[2] = -1
        features[3] = -1
    else:        
        bomb_map, timers = (np.array(game_state["bombs"])[:,0], np.array(game_state["bombs"])[:,1])
        bomb_fields = [(x[0] + i, x[1]) for x in np.array(game_state["bombs"])[:,0] for i in range(-s.BOMB_POWER,s.BOMB_POWER + 1)] \
                + [(x[0], x[1] + i) for x in np.array(game_state["bombs"])[:,0] for i in range(-s.BOMB_POWER, s.BOMB_POWER + 1)]
        d = np.array([[x[0],x[1]] for x in np.array(game_state["bombs"])[:,0]]) - np.array(agent_pos)
        # far away bombs ar not that interesting, therefore we limit the feature space to 4 more than the
        # actual bomb range
        features[3] = np.min([s.BOMB_POWER + 4,np.sum(np.abs(d[np.argmin(np.sum(d**2, axis=1))]))])
        features[1] = int(list(agent_pos) in bomb_fields)
        features[2] =  timers[np.argmin(np.sum(d**2,axis=1))]
    
    # FEATURE[4]: Distance to closest enemy
    if not game_state["others"] or len(game_state["others"]) == 0:
        features[4] = -1
    else:        
        others_fields = np.array([[x[3][0], x[3][1]] for x in game_state["others"]])
        d = others_fields - np.array(agent_pos)
        min_dist = np.sum(np.abs(d[np.argmin(np.sum(d**2,axis=1))]))
        features[4] = np.min([14, min_dist])
    
    # FEATURE[5] closest crate distance
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

    # FEATURE[6]: bool if position is in explosion
    explosions = []
    for i in range(1,16):
        for j in range(1,16):
            if game_state["explosion_map"][i,j] != 0:
                explosions += [[i,j]]

    if list(agent_pos) in explosions:
        features[6] = 1
    
    # FEATURE[7] (closest) walkable safe spot
    # FEATURE[8] bool if closest safe spot is reachable within bomb timer
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

    # if a Coin was collected the closed coin distance should be zero
    # but as it was removed from the field it doesn't show up.
    # These little three line fix that error
    if events:
        if e.COIN_COLLECTED in events:
            features[0] = 0

    return features.astype(int)

def find_walkable_safe_spots(current_pos, tested, safe, bomb_fields, explosions, field, paths, path = []):
    """
    Function to find safe spots for the agent to go to, if there is a bomb nearby.
    Recursive algorithm which moves around the agent's position and checks if the
    field is safe.
    Furthermore stores the path to the safe spot, so that we can actually get a proper
    distance to the closest safe spot.

    :param current_pos: In the beginning the agent's position, when called recursivly the position to check
    :param tested: List of fields that were already tested for being safe or unsafe
    :param safe: List of fields to store safe fields in
    :param bomb_fields: to check for safety of a field, we need to know where bombs are placed
    :param explosions: also fields that are currently filled with an explosions are unsafe
    :param field: field to check whether a movement direction is legal or illegal
    :param paths: dict of paths to later acces the safe spot with its path again
    :param path: the current path taken in recursive step
    """

    # copy path so it becomes the current path for the iterative step
    path = copy.deepcopy(path)
    # Add current pos to path
    path += [current_pos]

    # as the function technically only checks one path, and not necessarily
    # the shortest path, the if statement here shouldn't be necessary
    if tuple(current_pos) in paths.keys():
        paths[tuple(current_pos)] += path
    else:
        paths[tuple(current_pos)] = path

    # Add current field to tested fields 
    tested += [current_pos]

    # Add field to safe fields if it seems to be safe
    if is_safe(current_pos,bomb_fields, explosions):
        safe += [current_pos] 

    # Walk around if not an illegal move and field was not tested before
    if field[current_pos[0] - 1, current_pos[1]] == 0 and [current_pos[0] - 1, current_pos[1]] not in tested:
        find_walkable_safe_spots([current_pos[0]-1, current_pos[1]], tested,safe,bomb_fields,explosions,field,paths,path)
    if field[current_pos[0] + 1, current_pos[1]] == 0 and [current_pos[0] + 1, current_pos[1]] not in tested:
        find_walkable_safe_spots([current_pos[0]+1, current_pos[1]], tested,safe,bomb_fields,explosions,field,paths,path)
    if field[current_pos[0], current_pos[1] - 1] == 0 and [current_pos[0], current_pos[1] - 1] not in tested:
        find_walkable_safe_spots([current_pos[0], current_pos[1]-1], tested,safe,bomb_fields,explosions,field,paths,path)
    if field[current_pos[0], current_pos[1] + 1] == 0 and [current_pos[0], current_pos[1] + 1] not in tested:
        find_walkable_safe_spots([current_pos[0], current_pos[1]+1], tested,safe,bomb_fields,explosions,field,paths,path)    


def is_safe(current_pos,bomb_fields,explosions):
    """
    Function to check whether a given field is safe, meaning it is not within a bomb's range and
    moreover not a field where there is currently an explosion

    :param current_pos: Field to check
    :param bomb_fields: List of bombs to check against
    :param explosions: List of fields that are currently part of an explosion
    """
    if tuple(current_pos) not in bomb_fields and list(current_pos) not in explosions:
        return True
