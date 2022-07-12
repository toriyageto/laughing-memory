from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend()

lin_reg = LinearRegression()
X=4*np.random.rand(100,1)
y=4+3*X+np.random.rand(100,1)
# print(y)
# print(y.ravel()) #相当于y.reshape(-1),类似于矩阵铺平操作

lin_reg.fit(X,y)
X_new=[[0],[2],[4]]
# print(lin_reg.intercept_,lin_reg.coef_)
y_predict=lin_reg.predict(X_new)
# print(lin_reg.predict(X_new)) #对y=f(x)这个函数带入x=0，2，4分别求y

# 数据拟合
# plt.plot(X_new,y_predict,"r-")
# plt.plot(X,y,"b.")
# plt.axis([0,4,0,20])
# plt.show()

# 随机梯度下降
sgd_reg=SGDRegressor(n_iter_no_change=50,penalty=None,eta0=0.1)
sgd_reg.fit(X,y.ravel())
X_new2=[[0],[2],[4]]
sgd_predict=sgd_reg.predict(X_new2)
# print(sgd_reg.intercept_,sgd_reg.coef_)
# plt.plot(X_new,sgd_predict,"r-")
# # plt.plot(X_new,y_predict,"b-")
# plt.plot(X,y,"b.")
# plt.axis([0,4,0,20])
# plt.show()

# 多项式回归
# X2=6*np.random.rand(1000,1)-3
# y2=0.5*X2**2+X2+2+np.random.rand(1000,1)
# poly_features = PolynomialFeatures(degree=2,include_bias=False)
# X_poly = poly_features.fit_transform(X2) #X_poly现在包含原始特征[X]并加上了这个特征的平方[X^2]
# # print(X_poly[0])
# lin_reg2=LinearRegression()
# lin_reg2.fit(X_poly,y2)
# # lin_reg2.fit(X2,y2)
# # X2_new=[[-3],[0],[3]]
# # y2_predict=lin_reg2.predict(X2_new)
# y2_predict = np.dot(X_poly, lin_reg2.coef_.T) + lin_reg2.intercept_
# # print(y2_predict)
# plt.plot(X2,y2_predict,'b.')
# print(lin_reg2.intercept_,lin_reg2.coef_.ravel())#[2.50036538] [[0.98878852 0.49757751]],即y=2.5+0.50*X**2+0.99*X

# plt.plot(X2_new,y2_predict,'r-')
# plt.scatter(X2,y2)
# plt.show()

# plot_learning_curves(lin_reg2, X2, y2)
# plt.show()

# 降低模型的过拟合的好方法是正则化这个模型（即限制它）：模型有越少的自由度，就越难以拟合数据。

# 岭回归
# ridge_reg=Ridge(alpha=2,solver='cholesky')
# ridge_reg.fit(X_poly,y2)
# y3_predict=ridge_reg.predict(X2_new)
# plt.plot(X2_new,y3_predict,'r-')
# plt.scatter(X2,y2)
# plot_learning_curves(ridge_reg, X2, y2)
# plt.show()

# 决策边界(二分类器）
iris=datasets.load_iris()
# print(iris.keys())
X=iris['data'][:,3:] #petal width
y=(iris['target']==2).astype(int) #True=1,False=0
log_reg=LogisticRegression()
log_reg.fit(X,y)
X_new=np.linspace(0,3,1000).reshape(-1,1)
# print(X_new.shape)
y_proba=log_reg.predict_proba(X_new)
# print(y_proba)
# plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
# plt.show()

#softmax回归（多分类器）
X = iris["data"][:, (2, 3)] # petal length, petal width

# print(X)
y = iris["target"]
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
softmax_reg.fit(X, y)
petal_width = np.linspace(0, 3, 1000).reshape(-1, 1).T
petal_length=np.linspace(1,7,1000).reshape(-1,1)
X_new=np.insert(petal_length,1,values=petal_width,axis=1)
y_proba = softmax_reg.predict_proba(X_new)
# print(y_proba)
