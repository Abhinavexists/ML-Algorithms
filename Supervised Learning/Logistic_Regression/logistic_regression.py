import numpy as np

np.random.seed(42)
x = np.linspace(0, 10, 100)
y = (x > 5).astype(int)

class LogisticRegression:
    def __init__(self, m: float = 0.0, b: float = 0.0, learning_rate: float = 0.01):
        self.m = m
        self.b = b
        self.learning_rate = learning_rate

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1/(1 + np.exp(-x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        linear = self.m * x + self.b
        return self.sigmoid(linear)
    
    def loss_function(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n: int = len(y)
        # 1e-9 added to avoid logrithm of zero
        binary_cross_entropy = (-1/n)*np.sum((y*np.log(y_pred + 1e-9))+((1-y)*np.log(1-y_pred + 1e-9)))
        return binary_cross_entropy
    
    def gradient_descent(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        n = len(y)
        dm = (1/n)*np.dot((y_pred-y), x)
        db = (1/n)*np.sum(y_pred-y)

        self.m = self.m - self.learning_rate * dm
        self.b = self.b - self.learning_rate * db

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 100):
        loss = None
        for i in range(epochs):
            y_pred = self.predict(x)
            loss = self.loss_function(y, y_pred)
            self.gradient_descent(x, y, y_pred)
        return f"loss during training is {loss}"
        
model = LogisticRegression(10, 3, 0.01)
result = model.train(x, y, 1000)
print(result)
