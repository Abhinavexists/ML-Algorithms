import numpy as np

class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.mean = {}
        self.variance = {}
        self.classes = None

    def decision_fit(self, x:np.ndarray, y:np.ndarray):
        n_samples, n_features = x.shape # No of samples and features, features are not used here
        self.classes = np.unique(y)

        for c in self.classes:
            x_class = x[y == c]
            self.class_priors[c] = x_class.shape[0] / n_samples
            self.mean[c] = np.mean(x_class, axis=0)
            self.variance[c] = np.var(x_class, axis=0)

    def gaussian_probability(self, class_label: int, x: np.ndarray) -> np.ndarray:
        mean = self.mean[class_label]
        variance = self.variance[class_label]
        exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
        return (1 / np.sqrt(2 * np.pi * variance)) * exponent
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        y_pred = []
        for sample in x:
            class_probabilities = {}
            for c in self.classes:
                prior = np.log(self.class_priors[c])
                likelihood = np.sum(np.log(self.gaussian_probability(c, sample)))
                class_probabilities[c] = prior + likelihood
            predicted_class = max(class_probabilities, key=class_probabilities.get)
            y_pred.append(predicted_class)
        return np.array(y_pred)
    

x = np.array([[1.0, 2.1], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0], [1.2, 0.9], [5.5, 8.5]])
y = np.array([0, 0, 1, 1, 0, 1])

model = NaiveBayes()
model.decision_fit(x, y)
predictions = model.predict(x)

print("Predictions:", predictions)
print("Means:", model.mean)
print("Variances:", model.variance)
print("Class Priors:", model.class_priors)
