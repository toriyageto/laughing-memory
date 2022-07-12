import pandas as pd
from keras.datasets import boston_housing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf

(x_train,y_train),(x_test,y_test)=boston_housing.load_data()
# print(x_train.shape)
# print(y_train.shape)

classes=['CRIM','ZN','INUDS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']

data=pd.DataFrame(x_train,columns=classes)
# print(data)
data['MEDV']=pd.Series(data=y_train)
# print(data)

# sns.pairplot(data[['MEDV','CRIM','AGE','DIS','TAX']],diag_kind='kde') #特征关系表
# g=sns.PairGrid(data[['MEDV','CRIM','AGE','DIS','TAX']])
# g.map_diag(sns.kdeplot)
# g.map_offdiag(sns.kdeplot,cmap='Blues_d',n_levels=6) #6等分的等高线
# plt.show()

scaler=preprocessing.MinMaxScaler()  #数据调整到0-1
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)
# print(x_train.shape)

# model=tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(units=32,activation=tf.nn.relu,input_dim=x_train.shape[1]))
# model.add(tf.keras.layers.Dense(units=64,activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=1))
# model.compile(optimizer='sgd',loss='mse',metrics=['mae']) #回归优化算法
# history=model.fit(x=x_train,y=y_train,epochs=10000,batch_size=len(y_train))

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=320,activation=tf.nn.relu,input_dim=x_train.shape[1]))
model.add(tf.keras.layers.Dense(units=640,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=640,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=1))
learning_rate=0.001
opt1=tf.keras.optimizers.Nadam(lr=learning_rate)
model.compile(optimizer=opt1,loss='mse',metrics=['mae']) #回归优化算法
history=model.fit(x=x_train,y=y_train,epochs=10000,batch_size=len(y_train))


score=model.evaluate(x_test,y_test)
print(score)
# [32.17009353637695, 4.550229549407959] cost花费为4.55，有4.55*1000的误差
# [33.12302780151367, 4.770251274108887]
y_pred=model.predict(x_test)
print(y_pred[:10])
print(y_test[:10])

plt.plot(history.history['mae'])
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train mae'],loc='upper right')
plt.show()