import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

def lrelu(x):
    if (x >= 0):
        return x
    else:
        return 0.01 * x

def derivate_lrelu(x):
    if (x >= 0):
        return 1
    else:
        return 0.01

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivate_sigmoid(x):
    f = sigmoid(x)
    return f * (1 - f)

def mse_count(y_true, y_predicted):
    return ((y_true - y_predicted) ** 2).mean()

class MultiLayerPerceptron:

    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.w7 = np.random.normal()
        self.w8 = np.random.normal()
        self.w9 = np.random.normal()
        self.w10 = np.random.normal()
        self.w11 = np.random.normal()
        self.w12 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        self.b4 = np.random.normal()
        self.b5 = np.random.normal()

        self.mses = []

    def predict(self, X):
        predicted = []
        for x in X:
            y_predicted = self.forward(x)
            if (y_predicted >= 0.5):
                predicted.append(1)
            else:
                predicted.append(0)
        return predicted

    def forward(self, x):
        h1 = lrelu(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1)
        h2 = lrelu(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2)
        h3 = lrelu(self.w7 * h1 + self.w8 * h2 + self.b3)
        h4 = lrelu(self.w9 * h1 + self.w10 * h2 + self.b4)
        o = sigmoid(self.w11 * h3 + self.w12 * h4 + self.b5)
        return o

    def train(self, data, y, x_test, y_test):
        rate = 0.05
        epoch = 100

        for i in range (epoch):
            for x, y_true in zip(data, y):
                h1_sum = self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1
                h1 = lrelu(h1_sum)
                h2_sum = self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2
                h2 = lrelu(h2_sum)
                h3_sum = self.w7 * h1 + self.w8 * h2 + self.b3
                h3 = lrelu(h3_sum)
                h4_sum = self.w9 * h1 + self.w10 * h2 + self.b4
                h4 = lrelu(h4_sum)
                o_sum = self.w11 * h3 + self.w12 * h4 + self.b5
                o = sigmoid(o_sum)

                y_predicted = o

                derivate_mse_by_y_predicted = -2 * (y_true - y_predicted)

                derivate_y_predicted_by_w11 = h3 * derivate_sigmoid(o_sum)
                derivate_y_predicted_by_w12 = h4 * derivate_sigmoid(o_sum)
                derivate_y_predicted_by_b5 = derivate_sigmoid(o_sum)
                derivate_y_predicted_by_h3 = self.w11 * derivate_sigmoid(o_sum)
                derivate_y_predicted_by_h4 = self.w12 * derivate_sigmoid(o_sum)

                derivate_h3_by_w7 = h1 * derivate_lrelu(h3_sum)
                derivate_h3_by_w8 = h2 * derivate_lrelu(h3_sum)
                derivate_h3_by_b3 = derivate_lrelu(h3_sum)
                derivate_h3_by_h1 = self.w7 * derivate_lrelu(h3_sum)
                derivate_h3_by_h2 = self.w8 * derivate_lrelu(h3_sum)

                derivate_h4_by_w9 = h1 * derivate_lrelu(h4_sum)
                derivate_h4_by_w10 = h2 * derivate_lrelu(h4_sum)
                derivate_h4_by_b4 = derivate_lrelu(h4_sum)
                derivate_h4_by_h1 = self.w9 * derivate_lrelu(h4_sum)
                derivate_h4_by_h2 = self.w10 * derivate_lrelu(h4_sum)

                derivate_h1_by_w1 = x[0] * derivate_lrelu(h1_sum)
                derivate_h1_by_w2 = x[1] * derivate_lrelu(h1_sum)
                derivate_h1_by_w3 = x[2] * derivate_lrelu(h1_sum)
                derivate_h1_by_b1 = derivate_lrelu(h1_sum)

                derivate_h2_by_w4 = x[0] * derivate_lrelu(h2_sum)
                derivate_h2_by_w5 = x[1] * derivate_lrelu(h2_sum)
                derivate_h2_by_w6 = x[2] * derivate_lrelu(h2_sum)
                derivate_h2_by_b2 = derivate_lrelu(h2_sum)

                self.w11 -= rate * derivate_mse_by_y_predicted * derivate_y_predicted_by_w11
                self.w12 -= rate * derivate_mse_by_y_predicted * derivate_y_predicted_by_w12
                self.b5 -= rate * derivate_mse_by_y_predicted * derivate_y_predicted_by_b5

                self.w7 -= rate * derivate_mse_by_y_predicted * derivate_y_predicted_by_h3 * derivate_h3_by_w7
                self.w8 -= rate * derivate_mse_by_y_predicted * derivate_y_predicted_by_h3 * derivate_h3_by_w8
                self.b3 -= rate * derivate_mse_by_y_predicted * derivate_y_predicted_by_h3 * derivate_h3_by_b3

                self.w9 -= rate * derivate_mse_by_y_predicted * derivate_y_predicted_by_h4 * derivate_h4_by_w9
                self.w10 -= rate * derivate_mse_by_y_predicted * derivate_y_predicted_by_h4 * derivate_h4_by_w10
                self.b4 -= rate * derivate_mse_by_y_predicted * derivate_y_predicted_by_h4 * derivate_h4_by_b4

                self.w1 -= rate * derivate_mse_by_y_predicted * (derivate_y_predicted_by_h3 * derivate_h3_by_h1 + derivate_y_predicted_by_h4 * derivate_h4_by_h1) * derivate_h1_by_w1
                self.w2 -= rate * derivate_mse_by_y_predicted * (derivate_y_predicted_by_h3 * derivate_h3_by_h1 + derivate_y_predicted_by_h4 * derivate_h4_by_h1) * derivate_h1_by_w2
                self.w3 -= rate * derivate_mse_by_y_predicted * (derivate_y_predicted_by_h3 * derivate_h3_by_h1 + derivate_y_predicted_by_h4 * derivate_h4_by_h1) * derivate_h1_by_w3
                self.b1 -= rate * derivate_mse_by_y_predicted * (derivate_y_predicted_by_h3 * derivate_h3_by_h1 + derivate_y_predicted_by_h4 * derivate_h4_by_h1) * derivate_h1_by_b1

                self.w4 -= rate * derivate_mse_by_y_predicted * (derivate_y_predicted_by_h3 * derivate_h3_by_h2 + derivate_y_predicted_by_h4 * derivate_h4_by_h2) * derivate_h2_by_w4
                self.w5 -= rate * derivate_mse_by_y_predicted * (derivate_y_predicted_by_h3 * derivate_h3_by_h2 + derivate_y_predicted_by_h4 * derivate_h4_by_h2) * derivate_h2_by_w5
                self.w6 -= rate * derivate_mse_by_y_predicted * (derivate_y_predicted_by_h3 * derivate_h3_by_h2 + derivate_y_predicted_by_h4 * derivate_h4_by_h2) * derivate_h2_by_w6
                self.b2 -= rate * derivate_mse_by_y_predicted * (derivate_y_predicted_by_h3 * derivate_h3_by_h2 + derivate_y_predicted_by_h4 * derivate_h4_by_h2) * derivate_h2_by_b2
            self.mses.append(self.test(x_test, y_test))

    def test(self, data, y):
        y_predicteds = np.apply_along_axis(self.forward, 1, data)
        mse = mse_count(y, y_predicteds)
        return mse

    def get_errors(self, data, y):
        error_count = 0
        y_predicted = self.predict(data)
        for i in range (len(y)):
            if (y[i] != y_predicted[i]):
                error_count += 1
        return error_count

database = pd.read_csv('covtype.data', header = None)
database = database.sort_values(by = 54)
x = database.iloc[543136:, [1, 2, 3, 4, 5, 6, 7, 8, 9]].values
y = database.iloc[543136:, [-1]].values
y = np.where(y == 6, 0, 1)
y = y.ravel()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

pca = PCA(n_components = 3)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

mlp = MultiLayerPerceptron()
mlp.train(x_train, y_train, x_test, y_test)
print('MSE по собственной нейронной сети: ', mlp.test(x_test, y_test))
print('Процент ошибки по собственной нейронной сети: ', mlp.get_errors(x_test, y_test) / len(x_test) * 100)
plt.plot(mlp.mses)
plt.title('График изменения MSE по эпохам')
plt.show()

print('\nПроверка нейронной сети через MLPClassifier:')
classifier = MLPClassifier(max_iter = 1000)
classifier.fit(x_train, y_train)
classifier.predict(x_test)
y_predicted = classifier.predict(x_test)
print('MSE в MLPClassifier: ', metrics.mean_squared_error(y_test, y_predicted))
error_count = 0
for i in range (len(y_test)):
    if (y_test[i] != y_predicted[i]):
        error_count += 1
print('Процент ошибок по MLPClassifier: ', error_count / len(y_test) * 100)