import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

class Perceptron:
    def __init__(self,learning_rate=0.01, iterateion=1000):
        self.lr=learning_rate
        self.iter=iterateion
        self.weights=None
        self.bias=None

    def fit(self,X,y):
        n_sample,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        for _ in range(self.iter):
            for idx, x_i in enumerate(X):
                output=np.dot(x_i,self.weights)+self.bias
                y_predicted = self._activation(output)
                update=self.lr*(y[idx]-y_predicted)
                self.weights+=update*x_i
                self.bias+=update


    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation(linear_output)

    def _activation(self, x):
        return np.where(x >= 0, 1, 0)

def main():
    # Load the dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    X = X[y != 2]  # Binary classification
    y = y[y != 2]  # Only classes 0 and 1

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Initialize and train the Perceptron model
    p = Perceptron(learning_rate=0.01, iterateion=1000)
    p.fit(X_train, y_train)

    # Evaluate the model
    predictions = p.predict(X_test)
    print("Perceptron classification accuracy:", accuracy_score(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    main()
     
