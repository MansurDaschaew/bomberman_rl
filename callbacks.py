#from _typeshed import IdentityFunction
from dataclasses import field
import os
import pickle
import random
from webbrowser import get

import numpy as np
import scipy.special
import heapq


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
        weights = np.random.rand(len(ACTIONS), get_num_features())
        self.model = weights / weights.sum()
        #self.model = np.zeros((weights/weights.sum(), 8))
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)



# choose a eps greedy action
#def eps_greedy(q_values, eps): 
#    if np.random.random() < eps:
#        return np.random.choice(len(q_values))
#    else:
#        return np.argmax(q_values)

#softmax 
#def softmax(q_values, temp): 
#    prob = scipy.special.softmax(q_values / temp)
#    return np.random.choice(len(q_values, p=prob))

#choose a action according the desired policy
def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    action_map = game_state #first normalize game
    features = state_to_features(game_state)
    #q_values = self.model
    q_values = np.matmul(self.model, features)
    #q_values = np.dot(self.model, features)
    
    #eps_greedy(q_values, eps)
    #eps greedy: todo Exploration vs exploitation
    if self.train: 
        if game_state["round"] == 1:
            eps = 1
        else: 
            eps = 0.01 + 0.1*np.exp(-game_state["round"]/200)
    else: 
        eps = 0.1

    if np.random.random() >= eps: 
        #a = 1 #define a = Q_t(a)
        #action = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'][np.argmax(a)]
        #action = np.argmax(q_values)
        #p = np.array([1,1,1,1,0.1,0.1])
        #p /= np.sum(p)
        #action = np.random.choice(['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'], p=p)
        p = np.argmax(q_values)
    
    else:
        #p = np.array([1,1,1,1,0,0])
        #p /= np.sum(p)
        p = np.random.choice(len(q_values))#
        #action = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'][p]
        #action = np.argmax(q_values)
    #action = np.random.choice(ACTIONS, p=[0.25, 0.25, 0.25, 0.25, 0.0, 0.0])

    action = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'][p]
    self.logger.info("Pick %s"%action)

    return action

#train /setup 
#if __name__ == '__main__':
#    env = gym.make(’CartPole-v0’)
#    alpha = 0.1 #rate 
#    gamma = 0.9 #discout factor
#    eps = 1.0
#
#    state = []
#    for i in range(len(cartPosSpace)+1):
#        for j in range(len(cartVelSpace)+1):
#            for k in range(len(cartThetaSpace)+1):
#                for l in range(len(cartThetaVelSpace)+1):
#                    states.append((i,j,k,l))
#
#    Q = {}
#    for s in states: 
#        for a in range(e):
#            Q[(s,a)] = 0.0
#    
#    n = 16 #steps looking back (n step Sarsa)
#    state_memory = np.zeros((n,4)) #states must be updated 
#    action_memory = np.zeros(n)
#    reward_memory = np.zeros(n)
#    reward_memory = np.zeros(n)
#
#    scores = []
#    n_episodes = 400 #how many steps are beeing played
#    for i in range(n_episodes): 
#        done = False 
#        score = 0.0
#        t = 0 
#        T = np.inf
#        observation = choose_action(Q, observation, eps)
#        action_memory[t%n] = action
#        state_memory[t%n] = observation
#
#        while not done:
#            observation, reward, done, info = env.step(action)
#            score += reward
#            state_memory[(t+1)%n] = observation 
#            reward_memory[(t+1)%n] = reward 
#            if done: 
#                T = t + 1
#        
#        action = choose_action(Q,observation,eps)
#        action_memory[(t+1)%n] = action 
#        tau = t - n - 1
#        if tau >= 0:
#            G = [gamma**(j-tau-1)*reward_memory[j%n] \
#                for j in range(tau+1, min(tau+n,T)+1)]
#            G = np.sum(G)
#            if tau + n < T:
#                s = get_state(state_memory[(tau+n%n)])
#                a = int(action_memory[(tau+n)%n])
#                G += gamma**n * Q([s,a])
#            s = get_state(state_memory[tau%n])
#            a = action_memory[tau%n]
#            Q[(s,a)] += alpha * (G-Q[(s,a)])
#
#    for tau in range(t-n+1,T):
#        G = [gamma**(j-tau-1)*reward_memory[j%n]\
#                for j in range(tau+1,min(tau+n, T)+1)]
#        G = np.sum(G)
#        if tau + n < T: 
#            s = get_state(state_memory[(tau+n%n)])
#            a = int(action_memory[(tau+n)%n])
#            G += gamma**n * Q([s,a])
#        
#        s = get_state(state_memory[tau%n])
#        a = action_memory[tau%n]
#        Q[(s,a)] += alpha * (G-Q[(s,a)])
#
#    scores.append(score)
#    avg_score = np.mean(score[-1000:])
#    epsilon = epsilon - 2 / n_episodes if epsilon > 0 else 0
#    if i % 1000 == 0: 
#        print(’episode ’, i, ’avg_score %.1f’ % avg_score, ’epsilon %.2f’ % epsilon)
#
#    
#    n = 16 #steps looking back (n step Sarsa)
#    state_memory = np.zeros((n,4)) #states must be updated 
#    action_memory = np.zeros(n)
#    reward_memory = np.zeros(n)
#    reward_memory = np.zeros(n)
#
#    scores = []
#    n_episodes = 400 #how many steps are beeing played
#    for i in range(n_episodes): 
#        done = False 
#        score = 0.0
#        t = 0 
#        T = np.inf
#        observation = choose_action(Q, observation, eps)
#        action_memory[t%n] = action
#        state_memory[t%n] = observation
#
#        while not done:
#            observation, reward, done, info = env.step(action)
#            score += reward
#            state_memory[(t+1)%n] = observation 
#            reward_memory[(t+1)%n] = reward 
#            if done: 
#                T = t + 1
#        
#        action = choose_action(Q,observation,eps)
#        action_memory[(t+1)%n] = action 
#        tau = t - n - 1
#        if tau >= 0:
#            G = [gamma**(j-tau-1)*reward_memory[j%n] \
#                for j in range(tau+1, min(tau+n,T)+1)]
#            G = np.sum(G)
#            if tau + n < T:
#                s = get_state(state_memory[(tau+n%n)])
#                a = int(action_memory[(tau+n)%n])
#                G += gamma**n * Q([s,a])
#            s = get_state(state_memory[tau%n])
#            a = action_memory[tau%n]
#            Q[(s,a)] += alpha * (G-Q[(s,a)])
#
#    for tau in range(t-n+1,T):
#        G = [gamma**(j-tau-1)*reward_memory[j%n]\
#                for j in range(tau+1,min(tau+n, T)+1)]
#        G = np.sum(G)
#        if tau + n < T: 
#            s = get_state(state_memory[(tau+n%n)])
#            a = int(action_memory[(tau+n)%n])
#            G += gamma**n * Q([s,a])
#        
#        s = get_state(state_memory[tau%n])
#        a = action_memory[tau%n]
#        Q[(s,a)] += alpha * (G-Q[(s,a)])
#
#    scores.append(score)
#    avg_score = np.mean(score[-1000:])
#    epsilon = epsilon - 2 / n_episodes if epsilon > 0 else 0
#    if i % 1000 == 0: 
#        print(’episode ’, i, ’avg_score %.1f’ % avg_score, ’epsilon %.2f’ % epsilon)

def get_free_tiles(field):
    """
    This function takes the field and returns all tiles where the agent could walk
    :param field: a 2D numpy array of the field
    :return: a 2D numpy array of type bool
    """

    free_tiles = np.zeros(field.shape, dtype=bool)
    for i in range(len(free_tiles)):
        for j in range(len(free_tiles[i])):
            if field[i, j] == 0:
                free_tiles[i, j] = 1

    return free_tiles

def get_neighbors(pos):
    # Position of t b r l neighbor
    npos = [(pos[0], pos[1] - 1), (pos[0], pos[1] + 1), (pos[0] + 1, pos[1]), (pos[0] - 1, pos[1])]
    return npos

def get_crates_list(field):
    return [(x, y) for x in range(17) for y in range(17) if field[x, y] == 1]
        
def get_first_step_on_path_features(path):
    features = np.array([0, 0, 0, 0])
    if path is not None and len(path) > 1:
        pos = path[0]
        next_pos = path[1]
        for neighbor_index, neighbor_pos in enumerate(get_neighbors(pos)):
            if next_pos == neighbor_pos:
                features[neighbor_index] = 1
    return features

def get_first_step_to_nearest_object_features(free_tiles, pos, objects, offset):
    ret = get_first_step_on_path_features(get_nearest_object_path(free_tiles, pos, objects, offset))
    return ret

# This function is needed to create relative maps from the agents position
def get_relative_maps(game_state):
    """
    This function takes the game_state and creates relative maps. The center (15, 15) is the agent's position
    :param game_state: a python dictionary that contains information about
        - the agent's position
        - the field (crates, walls, free tiles)
        - the position of the bombs and when the explode
        - a list of all coins
    :return: a dictionary with all the relative maps that will be needed
        'wall_map'
        'crate_map'
        'coin_map'
        'bomb_map_0'
        'bomb_map_1'
        'bomb_map_2'
        'bomb_map_3'
        'bomb_map_4'
    """

    # get attributes from game state
    self_x, self_y = game_state['self'][3]
    field = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']

    # create the dictionary of relative maps
    dict_keys = ['wall_map', 'crate_map', 'coin_map',
                 'bomb_map_0', 'bomb_map_1', 'bomb_map_2', 'bomb_map_3', 'bomb_map_4']

    relative_maps = dict()
    for key in dict_keys:
        relative_maps[key] = np.zeros((31, 31))

    # compute wall- and crate-maps
    for x in range(17):
        for y in range(17):
            x_rel, y_rel = x - self_x, y - self_y
            if field[x, y] == -1:
                relative_maps['wall_map'][15 + y_rel, 15 + x_rel] = 1
            elif field[x, y] == 1:
                relative_maps['crate_map'][15 + y_rel, 15 + x_rel] = 1

    # compute bomb-maps
    for bomb_pos, time in bombs:
        x, y = bomb_pos
        x_rel, y_rel = x - self_x, y - self_y
        relative_maps[f'bom_map_{time}'][15 + y_rel, 15 + x_rel] = 1

    # compute coin-map
    for x, y in coins:
        x_rel, y_rel = x - self_x, y - self_y
        relative_maps['coin_map'][15 + y_rel, 15 + x_rel] = 1

    return relative_maps

def restrict_relative_map(relative_map, radius):
    """
    This function takes a relative map and reduces the size of it to the radius
    :param relative_map: a 2D numpy array, calculated before with get_relative_map()
    :param radius: an integer value with the look around og the agent
    :return: a 2D numpy array of the reduced array
    """
    index_min = 15 - radius
    index_max = 16 + radius
    
    return relative_map[index_min:index_max, index_min:index_max]


"""
--------------------------------------------------------------------------------
shortest path features ---------------------------------------------------------
--------------------------------------------------------------------------------
"""


# For all shortest path features you need the Node class,
# because the A* search algorithm that is used, works on them:
class Node:
    """
    This class is needed to perform an A* search
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.g < other.g


# This algorithm is used to find the shortest path
# It only works if the class Node is defined
def shortest_path(free_tiles, start, target):
    """
    This is an A* search algorithm with heap queues to find the shortest path from start to target node
    :param free_tiles: free tiles is a 2D numpy array that contains TRUE if a tile is free or FALSE if not
    :param start: a (x,y) tuple with the position of the agent
    :param target: a (x,y) tuple with the position of the target
    :return: a list that contains the shortest path from start to target
    """

    start_node = Node(None, start)
    target_node = Node(None, target)
    start_node.g = start_node.h = start_node.f = 0
    target_node.g = target_node.h = target_node.f = 0

    open_nodes = []
    closed_nodes = []

    heapq.heappush(open_nodes, (start_node.f, start_node))
    free_tiles[target] = True

    while len(open_nodes) > 0:
        current_node = heapq.heappop(open_nodes)[1]
        closed_nodes.append(current_node)

        if current_node == target_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        # get the current node and all its neighbors
        neighbors = []
        i, j = current_node.position
        neighbors_pos = [(i, j) for (i, j) in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)] if free_tiles[i, j]]

        for position in neighbors_pos:
            new_node = Node(current_node, position)
            neighbors.append(new_node)

        for neighbor in neighbors:
            if neighbor in closed_nodes:
                continue

            neighbor.g = current_node.g + 1
            neighbor.h = ((neighbor.position[0] - target_node.position[0]) ** 2) + (
                    (neighbor.position[1] - target_node.position[1]) ** 2)
            neighbor.f = neighbor.g + neighbor.h

            if not any(node[1] == neighbor for node in open_nodes):
                heapq.heappush(open_nodes, (neighbor.f, neighbor))
                continue

            for open_node in open_nodes:
                if neighbor == open_node[1]:
                    open_node[1].f = min(neighbor.f, open_node[1].f)
                    break

    return [start]


# This function returns the path to the nearest object in the objects list
# This requires the implementation of the class Node and the shortest_path algorithm
def get_nearest_object_path(free_tiles, pos, objects, offset=0):
    """
    This function finds the path that need the fewest steps from
    the agents current_node position to the nearest object from the object list
    :param free_tiles: a 2D numpy array of type bool, the entries that are True are positions the agent can walk along
    :param pos: a (x,y) tuple with the agents position
    :param objects: a list of positions of objects
    :param offset: an integer that gives a value that will be added to the size of the environment
    around the agent where he is looking at objects
    :return: a list refers to the the path from the agent to the nearest object
    """

    min_path = None
    min_path_val = np.infty

    if len(objects) == 0:
        return [pos]

    best_dist = min(np.sum(np.abs(np.subtract(objects, pos)), axis=1))
    near_objects = [elem for elem in objects if np.abs(elem[0] - pos[0]) + np.abs(elem[1] - pos[1]) <= best_dist +
                    offset]

    if len(near_objects) == 0:
        return [pos]

    for elem in near_objects:
        path = shortest_path(free_tiles, pos, elem)
        len_path = len(path)
        if len_path < min_path_val:
            min_path_val = len_path
            min_path = path

    return min_path


#escape death
#unsafe tiles 

#features
def is_a_suicide(game_state):
    dang_coos = np.nonzero(game_state['explosion_map'] == 1)
    dang_coos.append(
    (np.where(game_state['bombs'][1] == 0)),
    (np.nonzero(game_state['explosion_map'] == 2)), 
    (np.where(game_state['bombs'][1] == 1)), 
    (np.where(game_state['bombs'][1] == 2)),
    (np.where(game_state['bombs'][1] == 3)))
    return dang_coos
#-----------------------------    
def get_unsafe_tiles(field, bombs):
    unsafe_positions = []
    for bomb_pos, _ in bombs:
        unsafe_positions.append(bomb_pos)

        for x_offset in range(1, 4):
            pos = (bomb_pos[0] + x_offset, bomb_pos[1])
            if pos[0] > 16 or field[pos] == -1:
                break
            unsafe_positions.extend(x for x in [pos] if x not in unsafe_positions)

        for x_offset in range(-1, -4, -1):
            pos = (bomb_pos[0] + x_offset, bomb_pos[1])
            if pos[0] < 0 or field[pos] == -1:
                break
            unsafe_positions.extend(x for x in [pos] if x not in unsafe_positions)

        for y_offset in range(1, 4):
            pos = (bomb_pos[0], bomb_pos[1] + y_offset)
            if pos[0] > 16 or field[pos] == -1:
                break
            unsafe_positions.extend(x for x in [pos] if x not in unsafe_positions)

        for y_offset in range(-1, -4, -1):
            pos = (bomb_pos[0], bomb_pos[1] + y_offset)
            if pos[0] < 0 or field[pos] == -1:
                break
            unsafe_positions.extend(x for x in [pos] if x not in unsafe_positions)

    return unsafe_positions


def get_reachable_tiles(pos, num_steps, field):
    if num_steps == 0:
        return [pos]
    elif num_steps == 1:
        ret = [pos]
        pos_x, pos_y = pos

        for pos_update in [(pos_x + 1, pos_y), (pos_x - 1, pos_y), (pos_x, pos_y + 1), (pos_x, pos_y - 1)]:
            if 0 <= pos_update[0] <= 16 and 0 <= pos_update[1] <= 16 and field[pos_update] == 0:
                ret.append(pos_update)

        return ret
    else:
        candidates = get_reachable_tiles(pos, num_steps - 1, field)
        ret = []
        for pos in candidates:
            ret.extend(x for x in get_reachable_tiles(pos, 1, field) if x not in ret)
        return ret


def get_reachable_safe_tiles(pos, field, bombs, look_ahead=True):
    if len(bombs) == 0:
        raise ValueError("No bombs placed.")

    timer = bombs[0][1] if look_ahead else bombs[0][1] + 1
    reachable_tiles = set(get_reachable_tiles(pos, timer, field))
    unsafe_tiles = set(get_unsafe_tiles(field, bombs))

    return [pos for pos in reachable_tiles if pos not in unsafe_tiles]


def is_safe_death(pos, field, bombs, look_ahead=True):
    if len(bombs) == 0:
        return False

    return len(get_reachable_safe_tiles(pos, field, bombs, look_ahead)) == 0


def get_safe_death_features(pos, field, bombs):
    if len(bombs) == 0:
        return np.array([0, 0, 0, 0, 0])

    ret = np.array([], dtype=np.int32)
    for pos_update in [(pos[0], pos[1] - 1), (pos[0], pos[1] + 1), (pos[0] + 1, pos[1]), (pos[0] - 1, pos[1]), pos]:
        if field[pos_update] == 0:
            ret = np.append(ret, 1 if is_safe_death(pos_update, field, bombs) else 0)
        else:
            ret = np.append(ret, 1 if is_safe_death(pos, field, bombs) else 0)
    return ret


def is_bomb_suicide(pos, field):
    return is_safe_death(pos, field, [(pos, 3)], look_ahead=False)

#---------------------------
#---------------------------
# state to feature----------
#---------------------------
#---------------------------

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
    explosion_map = game_state["explosion_map"]
    #GET INFO OF PLAYING FIELD
    # get the feature funcs from from features.py 

    
    #free tiles
    free_tiles = get_free_tiles(field)
    
    crate_feat = get_first_step_to_nearest_object_features(free_tiles, pos, get_crates_list(field), 1)
    next_crate = np.array([len(get_crates_list(field)) == 2], dtype = np.int32)
    
    coin_feat = get_first_step_to_nearest_object_features(free_tiles, pos, coins, 2)

    #escape bomb and deadly koordinates
    #is_suicide = is_suicide(explosion_map)
    death = get_safe_death_features(pos, field, bombs)
    suicide = is_bomb_suicide(pos, field)
    place_bomb = game_state["self"][2]


    #features = np.array(crate_feat, next_crate, coin_feat, death, suicide, place_bomb)
    features = np.array([])
    features = np.append(features, crate_feat)
    features = np.append(features, next_crate)
    features = np.append(features, coin_feat)
    features = np.append(features, death)
    features = np.append(features, suicide)
    features = np.append(features, place_bomb)
    return features

#needed in setup for self.model
def get_num_features():
    dummy_state = {
        'round': 0,
        'step': 0,
        'field': np.zeros((17, 17)),
        'bombs': [],
        'explosion_map': np.zeros((17, 17)),
        'coins': [],
        'self': ("dummy", 0, True, (1, 1)),
        'others': []
    }

    return state_to_features(dummy_state).shape[0]