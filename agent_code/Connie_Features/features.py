

def features(game_state):
    features = np.zeros([9])
    
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    
    #boolean "can I place a bomb in the next step" -> monitor bombability
    bomb_action_possible = []
    (_,_,a,_) = game_state['self']
    bomb_action_possible.append(a)
    
    for i in range(game_state['round']):
        #if false occurs 8 times in a row -> returne TRUE
    
    # dangerous tiles in the next step (+ two steps later +three + four steps later)(-> monitor where not to go)
    
    dangerous_coordinates_next = np.nonzero(game_state['explosion_map'] == 1)
    dangerous_coordinates_next.append(np.where(game_state['bombs'][1] == 0))
    
    
    dangerous_coordinates_in_two = np.nonzero(game_state['explosion_map'] == 2)
    dangerous_coordinates_in_two.append(np.where(game_state['bombs'][1] == 1))
    
    dangerous_coordinates_in_three = np.where(game_state['bombs'][1] == 2)
    
    dangerous_coordinates_in_four = np.where(game_state['bombs'][1] == 3)
    
    ### WITH RADIUS?
    
    # safe tiles in the next step (with radius)
    ###safe_coordinates_next = np.where(game_state['explosion_map'] == 0)
    ###PLUS: everywhere where there s no bomb planted AND everywhere where bomb is not going to explode in next step
    
    
    
    
    
    # navigatable tiles positions (with a radius to minimize computationals effort) (-> know, where able to go)
    ## Find walkable directions -> left right up down
    agent_pos = game_state["self"][3]

    if agent_pos == (15,16) or agent_pos == (16,15):
        agent_pos = (15,15)

    ## As Wall is marked as -1 and Path 1, respectively, we cann add one (for now)
    ## To find out if we can walk in a direction or not
    ##if game_state["step"] == 1 or game_state["step"] == 2:
        ##print(agent_pos, game_state["field"][agent_pos])
    features[0] = game_state["field"][agent_pos[0]-1, agent_pos[1]] + 1
    features[1] = game_state["field"][agent_pos[0]+1, agent_pos[1]] + 1
    features[2] = game_state["field"][agent_pos[0], agent_pos[1] - 1] + 1
    features[3] = game_state["field"][agent_pos[0], agent_pos[1] + 1] + 1
    
    
    
    # nearest coins
    ## find closest coin
    closest_coin_pos = game_state["coins"][np.argmin(((game_state["coins"]-np.array(agent_pos))**2).sum(axis=1))]
    
    ## check if closest coin is in x or y direction
    x_or_y = np.argmax(np.abs(np.array(agent_pos) - np.array(closest_coin_pos)))
    
    
    
    
    
#? distance to bombs/next bomb

# position of bombable crates (as well with a radius to minimize computational effort)

def get_tiles(field):
    tiles = np.zeros(field.shape) - 1
    for i in range(len(tiles)):
        for j in range(tiles[i]):
            if field[i,j] == 1:
                tiles[i,j] = 1

    return tiles 

def get_crates(field):
    crates = []
    for i in range(len(field)):
        for j in range(field[i]):
            if field[i,j] == 1:
                crates.append([i,j])

    return crates 

## + see get nearest bomb tile
def get_crate_neighbors(pos):
    #top
    #bottow
    #left
    #right

    return

def get_nearest_bomb_tile(field, pos):
    x = get_nearest_tile(field, pos)[0]
    y = get_nearest_tile(field, pos)[1]
    #check if neighbours are 0
    all_tiles = get_tiles(field, pos)
    if all_tiles[x+1][y] == 0:
        r = np.linalg.norm(pos-(x+1,y))
    if all_tiles[x-1][y] == 0:
        l = np.linalg.norm(pos-(x-1,y))
    if all_tiles[x][y+1] == 0:
        u = np.linalg.norm(pos-(x,y+1))
    if all_tiles[x][y-1] == 0:
        d = np.linalg.norm(pos-(x,y-1))
    #see which pos is nearest  ...
    #for i in range(len())

# number of opponents alive (-> monitor danger of others)
def get_opponents_number(get_state_for_agent):
    return len(get_state_for_agent[others])

# number of rounds (-> monitor game time correlated actions)
def get_current_round(get_state_for_agent):
    return get_state_for_agent[round]


# position of opponent(s) + boolean if they can or cannot place a new bomb in the next step
def get_opponents_position():
    

# count of current points (maybe devide into two features: count of collected coins & count of destroyed opponents)

# count of overall points gained

# shortest path to nearest object
##see Mansurs features


