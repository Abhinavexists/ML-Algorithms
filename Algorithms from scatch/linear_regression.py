# Linear Regression from scratch 
# y = mx + b

import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])

# x = np.random.randint(1, 100, size=10)
# y = np.random.randint(1,100, size=10)

class LinearRegression:
    def __init__(self, m: float = 0.0, b: float = 0.0, learning_rate: float = 0.01):
        self.m = m # Slope
        self.b = b # intercept
        self.learning_rate = learning_rate

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.m * x + self.b
        
    def loss_function(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        n = len(y)
        mse = (1/n)*np.sum((y-y_pred)**2)
        return mse
    
    def gradient_descent(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        n = len(y)
        dm = (-2/n)*np.sum((y-y_pred)*x)
        db = (-2/n)*np.sum((y-y_pred))

        self.m = self.m - self.learning_rate*dm
        self.b = self.b - self.learning_rate*db

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 100):
        for i in range(epochs):
            y_pred = self.predict(x=x)
            loss = self.loss_function(y, y_pred)
            self.gradient_descent(x, y, y_pred)
            return f"loss during training is {loss}, the predicted values are {y_pred}"

model = LinearRegression(m=10, b=3, learning_rate=0.1)
# y_pred = np.array([model.predict(xi) for xi in x])
# loss = model.loss_function(y, y_pred=y_pred)
result = model.train(x, y, epochs=1000)

print(result)
