#------------------1---------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("./ToyotaCorolla.csv")
X = data['KM']
Y = data['Price']

plt.scatter(X,Y)
plt.xlabel('KM')
plt.ylabel('Price')
plt.title('Scatter Plot')
plt.show()



import random

# Define the objective function
def objective_function(x):
    return -x**2 + 5

# Hill Climbing Algorithm
def hill_climb(max_iterations=1000, step_size=0.1):
    # Start at a random point
    current_x = random.uniform(-10, 10)
    current_value = objective_function(current_x)

    for i in range(max_iterations):
        # Generate a neighboring solution
        neighbor_x = current_x + random.uniform(-step_size, step_size)
        neighbor_value = objective_function(neighbor_x)

        # If the neighbor is better, move to it
        if neighbor_value > current_value:
            current_x = neighbor_x
            current_value = neighbor_value

    return current_x, current_value

# Run the algorithm
best_x, best_value = hill_climb()

print("Best solution found:")
print(f"x = {best_x:.4f}")
print(f"f(x) = {best_value:.4f}")