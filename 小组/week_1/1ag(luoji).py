import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 定义 sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义逻辑回归类
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)

        costs = []
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)

            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 计算成本（可选，用于可视化训练过程）
            cost = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            costs.append(cost)

        # 可视化成本变化
        plt.plot(costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost over iterations')
        plt.show()

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_model)
        y_pred_cls = np.where(y_pred > 0.5, 1, 0)
        return y_pred_cls

# 加载鸢尾花数据集
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv(url, names=column_names)

# 将类别分为两类，0 代表山鸢尾，1 代表变色鸢尾和维吉尼亚鸢尾
data['species'] = data['species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 1})

X = data.drop('species', axis=1).values
y = data['species'].values

# 划分训练集和测试集
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    test_size = int(num_samples * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 创建逻辑回归模型实例
model = LogisticRegression(learning_rate=0.01, num_iterations=1000)

# 训练模型
model.fit(X_train, y_train)

# 进行预测
predictions = model.predict(X_test)

# 计算准确率
accuracy = np.mean(predictions == y_test)
print("准确率:", accuracy)