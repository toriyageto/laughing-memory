import tensorflow as tf
import numpy as np
from keras.losses import categorical_crossentropy


# 数据创建器
def CreateDatasets(high,iNum,iArraySize):
    x_train=np.random.random((iNum,iArraySize))*float(high)
    y_train=((x_train[:iNum,0]+x_train[:iNum,1])/2).astype(int)
    return x_train,y_train,tf.keras.utils.to_categorical(y_train,num_classes=high)

category=10 #y有10种标签
dim=2  #x的维度数据
x_train,y_train,y_train2=CreateDatasets(category,1000,dim)
# # 给所有[0,1]的数贴上0标签，给所有[1,2]的数贴上1标签
# x1=np.random.random((500,1)) #500个随机数，范围0-1
# x2=np.random.random((500,1))+1 #500个随机数，范围1-2
# x_train=np.concatenate((x1,x2))
# # print(x_train)
# y1=np.zeros((500,),dtype=int) #500个0标签
# y2=np.ones((500,),dtype=int) #500个1标签
# y_train=np.concatenate((y1,y2))
# # print(y_train)
# # 独热编码
# y_train2=tf.keras.utils.to_categorical(y_train,num_classes=2)
# # print(y_train2)

# 创建模型
# model=tf.keras.models.Sequential([
#     tf.keras.layers.Dense(10,activation=tf.nn.relu,input_dim=1), #输入层
#     tf.keras.layers.Dense(10,activation=tf.nn.relu), #Relu算法
#     tf.keras.layers.Dense(2,activation=tf.nn.softmax) #输出层
# ])

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.relu,input_dim=dim))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=category,activation=tf.nn.softmax))

# 编译
model.compile(optimizer='adam',loss=categorical_crossentropy,metrics=['accuracy'])

# 训练
model.fit(x=x_train,y=y_train2,epochs=200,batch_size=128)

# 评估正确率
# x_test=np.array([[0.22],[0.31],[1.22],[1.33]])
# y_test=np.array([0,0,1,1])
# y_test2=tf.keras.utils.to_categorical(y_test,num_classes=2)
x_test,y_test,y_test2=CreateDatasets(category,10,dim)
score=model.evaluate(x_test,y_test2,batch_size=128)
print(score)
# [0.341504842042923, 1.0] 准确率100%
predict=model.predict(x_test)
print(predict)
# [[0.5767596  0.42324036] 概率，即第一个数字0.22从输入层输进去，有57%的概率为0，42%的概率为1，以此类推
#  [0.5674147  0.43258527]
#  [0.4349016  0.5650984 ]
#  [0.4184966  0.58150333]]

#输入y标签的预测值和真实值
for i in range(len(x_test)):
    print(np.argmax(predict[i]),end=' ') #求出最大值的索引，0,0,1,1
print()
for i in range(len(y_test)):
    print(y_test[i],end=' ')