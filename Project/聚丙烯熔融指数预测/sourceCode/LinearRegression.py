import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error #均方误差

IO = "D:\大二上\软件技术基础\Project\聚丙烯熔融指数预测\data.xls"  #数据文件路径
data = pd.read_excel(io=IO)
features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'] 
x = data[features] # 定义参数列 
y = data['y'] 
y = data.y # 定义测试值列
value = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'y']

# 数据的描述统计
print(data[value].describe())

# 缺失值检验
print(data[value][data[value].isnull()==True].count())

# 数据相关性举证
print(data[value].corr())

# 数据可视化输出
paiplot=sn.pairplot(data, x_vars=['x1', 'x2' ,'x3' ,'x4' ,'x5' ,'x6' ,'x7' ,'x8' ,'x9'], y_vars='y', height=6, kind='reg')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=100, stratify=None)  #划分数据集


reg=LinearRegression()

#用训练集对模型进行训练
reg.fit(x_train,y_train)

#打印回归系数
print(reg.coef_)

#打印截距项
print('intercept =', reg.intercept_)

#模型的拟合优度
print('score =', reg.score(x_train,y_train)) 

#测试集的预测
y_predict=reg.predict(x_test)

#输出测试集mse
print('mse =', mean_squared_error(y_test,y_predict)) 

plt.figure(figsize=(12, 6))
plt.gca().set_facecolor('whitesmoke')
plt.plot(range(len(y_predict)), y_predict, '--o', color='gray', label='predict')
plt.plot(range(len(y_predict)), y_test, '-*', color='peru', label='test')
plt.title('LinearRegression', fontsize=18)
plt.xlabel('The Number of y', fontsize=13)
plt.ylabel('The Value of y', fontsize=13)
plt.grid(linestyle='--', linewidth=1, alpha=0.8)
plt.legend(facecolor='whitesmoke', fontsize=14)
plt.show()