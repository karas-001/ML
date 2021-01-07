from sklearn.datasets import load_iris # 导入鸢尾花数据集

from sklearn.linear_model import LogisticRegression # 导入决策树包

from sklearn.metrics import accuracy_score # 导入准确率评价指标

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


iris = load_iris() # 载入数据集
print('iris数据集特征')
print(iris.data[:10])

print('iris数据集标签')
print(iris.target)

clf = LogisticRegression(max_iter=5000) #加载模型  max_iter是设定的最大迭代次数



x_train, y_train = iris.data[:120], iris.target[:120]
x_dev,y_dev = iris.data[120:], iris.target[120:]
print(x_train[:5])
print(y_train)

clf.fit(x_train,y_train) # 模型训练，取前五分之四作训练集

predictions = clf.predict(x_dev) # 模型测试，取后五分之一作测试集
predictions[:10]
print(predictions)
print('Accuracy:%s'% accuracy_score(y_dev, predictions))


x = iris.data[:, :2]
y = iris.target
clf = LogisticRegression()
clf.fit(x, y.ravel())
N, M = 500, 500     # 横纵各采样多少个值
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()   # 第0列的范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()   # 第1列的范围
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
x1, x2 = np.meshgrid(t1, t2)                    # 生成网格采样点
x_test = np.stack((x1.flat, x2.flat), axis=1)   # 测试点

cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
y_hat = clf.predict(x_test)       # 预测值
y_hat = y_hat.reshape(x1.shape)                 # 使之与输入的形状相同
plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)     # 预测值的显示
plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), edgecolors='k', s=50, cmap=cm_dark)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid()
plt.savefig('2.png')
plt.show()

