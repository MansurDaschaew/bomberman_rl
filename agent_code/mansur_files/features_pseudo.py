from os import X_OK, path
from typing import Collection
import numpy as np 

#get info about tiles, 1 are destructable tiles, 0 are free tiles 
def get_tiles(field):
    tiles = np.zeros(field.shape) - 1
    for i in range(len(tiles)):
        for j in range(tiles[i]):
            if field[i,j] == 0:
                tiles[i,j] = 0
            elif field[i,j] == 1:
                tiles[i,j] = 1

    return tiles 

def get_neighbors(pos):
    #top
    #bottow
    #left
    #right

    return

#only nearest free tiles maybe add a get_nearest_object? 
def get_nearest_tile(field, pos):
    #get 
    all_tiles = get_tiles(field)
    #return as coordinated of the desired tile. If there are two or more solutions maybe choose the closest to the center?

#get nearest tile to bomb #pseudo data types must be adapted
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


#shortest path to nearest object, see breadth_first_search algorithm
def get_shortest_path(pos, destination):
    queue = collections.deque([[pos]])
    seen = set([pos])
    while queue:
        path = queue.popleft()
        x,y = path[-1]
        if destination[y][x] == goal:
            return path
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= x2 < width and 0 <= y2 < height and grid[y2][x2] != wall and (x2,y2)
    #...

def get_nearest_coin(pos, list_of_coins):
    indx_of_nearest_coin  = 0
    distance = 0 
    if list_of_coins == []:
        return None
    for i in range(len(list_of_coins)-1):
        dist = np.linalg.norm(pos-list_of_coins[i])
        if dist > distance: 
            pass
        else: 
            indx_of_nearest_coin = i
            distance = dist
    
    return list_of_coins[indx_of_nearest_coin]

def get_unsafe_tiles(pos, flield, bombs): 
    if len(bombs) == 0:
        return np.array([0,0,0,0])
    ret = np.array([], dtype=np.int32)
    for i in [pos[0],]    

def         