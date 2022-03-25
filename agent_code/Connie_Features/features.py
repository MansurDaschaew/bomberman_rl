



# boolean "can I place a bomb in the next step" -> monitor bombability

# dangerous tiles in the next step (+ two steps later +three + four steps later)(with a radius)(-> monitor where not to go)

#?same as above? safe tiles in the next step (with radius)
##see Mansurs features for code

# navigatable tiles positions (with a radius to minimize computationals effort) (-> know, where able to go)
##see Mansurs & Ulis features for code

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

# number of steps(-> monitor game time correlated actions)

# position of opponent(s) + boolean if they can or cannot place a new bomb in the next step

# count of current points (maybe devide into two features: count of collected coins & count of destroyed opponents)

# count of overall points gained

# shortest path to nearest object
##see Mansurs features

# nearest coins
##see Mansurs & Ulis features for code