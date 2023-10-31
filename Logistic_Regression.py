import pandas as pd
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def Logistic_Regression(X, Y, learning_rate, num_iterations):
    m, n = X.shape  # Number of training examples and features
    theta = np.zeros(n)  # Initialize model parameters

    for i in range(num_iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)

        gradient = np.dot(X.T, (h - Y)) / m

        theta -= learning_rate * gradient

    return theta

data = pd.read_csv("Preprocessed_dataset.csv")

data = pd.DataFrame(data)

X = np.array(data[['Landmark_1', 'Landmark_2', 'Landmark_3']])
Y_series = data['Class']
Y = pd.to_numeric(Y_series, errors='coerce').fillna(0).astype(int)


print("X.shape:", X.shape)
print("Y.shape:", Y.shape)

learning_rate = 0.01
num_iterations = 1000

theta = Logistic_Regression(X, Y, learning_rate, num_iterations) 

print(theta)