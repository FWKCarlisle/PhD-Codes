# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:33:48 2024

@author: ppysa5

Implementation of AStar path finding algorithm (https://en.wikipedia.org/wiki/A*_search_algorithm)
to a non-infinite, 2D, cartesian coordinate plane with obstacles.
Our plane (which here we call grid) is a 2D array, of 0s (allowed positions)
and 1s (obstacles ie. non-allowed positions).
"""

from astar import AStar # https://pypi.org/project/astar/. (Need to pip install astar)
import numpy as np


class AStarSquareGrid(AStar):
    """
    The AStar library is implemented here to find the most efficient path between 
    2 points on a 2D plane (of carteesian coordinates, ie. a square grid). The 
    implementation of the AStar library is a bit unconventional. From the AStar 
    documentation https://pypi.org/project/astar/ : << The astar module defines
    the AStar class, which has to be inherited from and completed with the 
    implementation of several methods. The functions take/return _node_ objects.>>
    The 4 methods below, serve to adapt the AStar class to our case scenario, 
    a non-infinite, 2D, cartesian coordinate plane. 
    """
    def __init__(self, grid):
        """
        The Astar library defines the (x,y) directions in complete opposite way
        to how Nanonis does so. So (xNanonis, yNanonis) = (yAstar, xAstar) relative
        to the scan orientation. To make the code more readable, I am going to 
        transpose of the space grid, rather than doing something like xNanonis = yAstar...
        """
        self.grid = grid.T
    
    def neighbors(self, node):
        """
        From the AStar documentation: <<For a given node, returns (or yields) 
        the list of its neighbors. This is the method that one would provide 
        in order to give to the algorithm the description of the graph to 
        use during for computation. This method must be implemented in a subclass.>>
        We have a grid of square nodes, ie. we have 4 neighbors per node located at
        (0, 1), (1, 0), (0, -1), (-1, 0). 
        """
        
        
        x, y = node
        neighbors = [(x + dx, y + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1,1), (-1,-1), (1,-1), (-1,1)]]
        """
        You can make the path only follow specific directions. Eg. for a path
        that only goes either right or up:
        neighbors = [(x + dx, y + dy) for dx, dy in [(0, 1), (1, 0)]]
        """
        
        # valid neighbors are those that are 1. within the bounds of our grid, and 2. equal to 0.
        return [(neighbor[0], neighbor[1]) for neighbor in neighbors if (0 <= neighbor[0] < len(self.grid)
                                                                         and 0 <= neighbor[1] < len(self.grid[0])
                                                                         and self.grid[neighbor[0], neighbor[1]] == 0)]



    def distance_between(self, n1, n2):
        """
        From the Astar documentation: <<Gives the real distance/cost between two 
        adjacent nodes n1 and n2 (i.e n2 belongs to the list of n1’s neighbors). 
        n2 is guaranteed to belong to the list returned by a call to neighbors(n1).
        This method must be implemented in a subclass.>>
        We consider all node edges to have the same cost. ie. this just needs to
        be set to any arbitrary constant eg. 1
        """
        return 1
    
    
    def heuristic_cost_estimate(self, current, goal):
        """
        From the Astar documentation: <<Computes the estimated (rough) 
        distance/cost between a node and the goal. The first argument is the 
        start node, or any node that have been returned by a call to the 
        neighbors() method. This method is used to give to the algorithm an hint 
        about the node he may try next during search. This method must be 
        implemented in a subclass.>> Here, a *rough* estimate of distance is
        valid. Thw "Manhattan" distance calculation is commonly used for pathfinding
        algorithms, so this is what I used below.
        """
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])
    
    def find_path(self, start, end):
        if self.astar(start, end):
            print ('path found')
            path = self.astar(start, end)
            x_path, y_path = zip(*path)
            return np.asarray(x_path), np.asarray(y_path)
        else: 
            print('no path found for start = ', start, 'end = ', end)
            return None, None


"""
# Example usage:

import matplotlib.pyplot as plt

grid = np.load("test_grid_with_obstacles.npy")

astar_grid = AStarSquareGrid(grid)

start = (25, 210)
end = (200, 50)

path_x, path_y = astar_grid.find_path(start, end)

fig, ax = plt.subplots()
ax.imshow(grid, alpha = 0.5)
ax.scatter(path_x, path_y, label = 'path')
ax.legend()

"""