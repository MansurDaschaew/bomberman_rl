

def features(game_state):
    #boolean "can I place a bomb in the next step" -> monitor bombability
    
    # dangerous tiles in the next step (+ two steps later +three + four steps later)(with a radius)(-> monitor where not to go
    
    dangerous_coordinates_next = np.nonzero(game_state['explosion_map'] == 1)
    dangerous_coordinates_next.append(np.where(game_state['bombs'][1] == 0))
    
    
    dangerous_coordinates_in_two = np.nonzero(game_state['explosion_map'] == 2)
    dangerous_coordinates_in_two.append(np.where(game_state['bombs'][1] == 1))
    
    dangerous_coordinates_in_three = np.where(game_state['bombs'][1] == 2)
    
    dangerous_coordinates_in_four = np.where(game_state['bombs'][1] == 3)
    
    # safe tiles in the next step (with radius)
    #safe_coordinates_next = np.where(game_state['explosion_map'] == 0)
    # PLUS: everywhere where there s no bomb planted AND everywhere where bomb is not gount to explode in next step


# navigatable tiles positions (with a radius to minimize computationals effort) (-> know, where able to go)
##see Mansurs & Ulis features for code
def get_navigatable_tiles(game_state):
    game_state['']
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

# nearest coins
##see Mansurs & Ulis features for code
