# --------11----------
import numpy as np

class Perceptron:
    def _init_(self): self.w = np.zeros(3)
    def predict(self, x): return 1 if np.dot(self.w, x) >= 0 else 0
    def train(self, X, y, lr=0.1, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                error = target - self.predict(xi)
                self.w += lr * error * xi

X = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])  # Bias + inputs

# AND
p_and = Perceptron()
p_and.train(X, [0,0,0,1])
print("AND:")
for x in X: print(f"{x[1:]} => {p_and.predict(x)}")

# OR
p_or = Perceptron()
p_or.train(X, [0,1,1,1])
print("\nOR:")
for x in X: print(f"{x[1:]} => {p_or.predict(x)}")