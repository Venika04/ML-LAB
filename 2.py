#-----------------2----------------------

# Visualize the n-dimensional data using 3D surface plots.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("ToyotaCorolla.csv")

#3d surface plot
x = data['KM']
y = data['Doors']
z = data['Price']

ax = plt.axes(projection='3d')
ax.plot_trisurf(x,y,z,cmap="jet")
ax.set_title("3D Surface Plot")

plt.show()





#Write a program to implement the Best First Search (BFS) algorithm.

def best_first_search(graph,start,goal,heuristic, path=[]):
    open_list = [(0,start)]
    closed_list = set()
    closed_list.add(start)

    while open_list:
        open_list.sort(key = lambda x: heuristic[x[1]], reverse=True)
        cost, node = open_list.pop()
        path.append(node)

        if node==goal:
            return cost, path

        closed_list.add(node)
        for neighbour, neighbour_cost in graph[node]:
            if neighbour not in closed_list:
                closed_list.add(neighbour)
                open_list.append((cost+neighbour_cost, neighbour))

    return None


graph = {
    'S': [('A', 2), ('B', 3)],
    'A': [('S', 2), ('C', 4)],
    'B': [('S', 3), ('C', 1)],
    'C': [('A', 4), ('B', 1), ('G', 2)],
    'G': []
}

heuristic = {
    'S': 5,
    'A': 3,
    'B': 2,
    'C': 1,
    'G': 0
}

start = 'S'
goal = 'G'

result = best_first_search(graph, start, goal, heuristic)

if result:
    print(f"Minimum cost path from {start} to {goal} is {result[1]}")
    print(f"Cost: {result[0]}")
else:
    print(f"No path from {start} to {goal}")