import numpy as np

class SVM:
    def __init__(self, regularization: float = 0.01, learning_rate: float = 0.01):
        self.regularization = regularization  
        self.learning_rate = learning_rate    
        self.w = None                         # weights (slope in higher dimensions)
        self.b = None                         # bias (intercept)

    def decision_boundary(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.w) + self.b

    def hinge_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        decision = np.dot(x, self.w) + self.b
        loss = np.maximum(0, 1 - y * decision)
        return np.mean(loss)

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 100):
        self.w = np.zeros(x.shape[1])  # initialize weights
        self.b = 0                     # initialize bias

        for i in range(int(epochs)):
            for i, xi in enumerate(x):
                condition = y[i] * (np.dot(xi, self.w) + self.b) >= 1

                if condition:
                    dw = 2 * self.regularization * self.w # L2 regularisatiom
                    db = 0
                else:
                    dw = 2 * self.regularization * self.w - y[i] * xi
                    db = -y[i]

                self.w = self.w - self.learning_rate * dw
                self.b = self.b - self.learning_rate * db
        
        return self.w, self.b

    def predict(self, x: np.ndarray) -> np.ndarray:
        decision = np.dot(x, self.w) + self.b
        return np.sign(decision)

x = np.array([[1], [2], [3], [4], [5], [6], [7]])   # shape (7,1)
y = np.array([1,1,1,-1,-1,-1,-1])
y = y.reshape(-1)

model = SVM()
result = model.train(x, y)
loss = model.hinge_loss(x, y)
print(result)
print(loss)
            
